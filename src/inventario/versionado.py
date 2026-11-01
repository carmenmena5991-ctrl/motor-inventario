"""Versionado de lo que se publica: modelo exportado, manifiesto del pedido y DVC.

Tres piezas, todas idempotentes:

1. `exportar_modelo` copia una versión del Model Registry de MLflow a
   models/<nombre>/v<versión>/. El pedido se genera con ESA copia, de modo que lo
   rastreado por DVC es exactamente el modelo que lo produjo y reproducirlo no depende
   del servidor MLflow.
2. `manifiesto_pedido` registra, junto a cada pedido, con qué se construyó: commit del
   código, versión y huella del modelo, huellas de los datos crudos y de las features y la
   del propio pedido. No lleva fecha de generación: re-ejecutar da el mismo archivo.
3. `versionar_con_dvc` guarda data/raw, data/processed, data/predictions y models en la
   caché de DVC (`dvc add`) y los sube al remoto (`dvc push`) si está accesible.

Los punteros .dvc que deja `dvc add` los confirma en git una persona, en una rama con Pull
Request (DECISIONES.md, sección 10).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from inventario.orquestacion import BASE_DEFAULT, Rutas

log = logging.getLogger(__name__)

RUTAS_VERSIONADAS = ("data/raw", "data/processed", "data/predictions", "models")
# Código que determina el resultado de un pedido (para marcar si difería del commit).
RUTAS_CODIGO = ("src", "dags", "config", "params.yaml")


def resolver_base(base: str | Path | None) -> Path:
    return Path(base or os.environ.get("INVENTARIO_DIR", BASE_DEFAULT))


def resolver_repo(repo: str | Path | None = None) -> Path:
    """Raíz del repositorio (punteros .dvc, .dvc/, .git): `repo`, INVENTARIO_REPO o la base.

    En Docker es /opt/airflow/repo, el repo montado como carpeta; en el host coincide con la
    base. Los datos son los mismos archivos vistos desde dos rutas.
    """
    return Path(repo or os.environ.get("INVENTARIO_REPO") or resolver_base(None))


def md5_archivo(ruta: str | Path) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def md5_directorio(ruta: str | Path) -> str:
    """Huella de un directorio: md5 de la lista ordenada (ruta relativa, md5 del archivo).
    No depende del orden de creación ni de fechas de modificación."""
    ruta = Path(ruta)
    if not ruta.is_dir():
        raise FileNotFoundError(f"No existe el directorio {ruta}")
    h = hashlib.md5()
    for archivo in sorted(p for p in ruta.rglob("*") if p.is_file()):
        h.update(f"{archivo.relative_to(ruta).as_posix()}\0{md5_archivo(archivo)}\n".encode())
    return h.hexdigest()


def ruta_modelo_local(nombre: str, version: str | int, base: str | Path | None = None) -> Path:
    return resolver_base(base) / "models" / nombre / f"v{version}"


def exportar_modelo(
    version: str | int,
    nombre: str | None = None,
    base: str | Path | None = None,
    tracking_uri: str | None = None,
) -> Path:
    """Descarga la versión `version` del Model Registry a models/<nombre>/v<versión>/.

    Si ya está exportada no se vuelve a descargar (una versión registrada no cambia). La
    descarga va a un directorio temporal que se renombra al final, para no dejar una
    exportación a medias si la tarea se interrumpe.
    """
    import mlflow

    from inventario.entrenar import MODELO_REGISTRADO_DEFAULT, resolver_tracking_uri

    nombre = nombre or MODELO_REGISTRADO_DEFAULT
    destino = ruta_modelo_local(nombre, version, base)
    if (destino / "MLmodel").exists():
        return destino

    temporal = destino.with_name(destino.name + ".tmp")
    shutil.rmtree(temporal, ignore_errors=True)
    temporal.mkdir(parents=True)
    mlflow.set_tracking_uri(resolver_tracking_uri(tracking_uri))
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{nombre}/{version}", dst_path=str(temporal))
    if not (temporal / "MLmodel").exists():
        raise RuntimeError(f"La exportación de {nombre} v{version} no contiene MLmodel")
    shutil.rmtree(destino, ignore_errors=True)
    temporal.rename(destino)
    return destino


def info_codigo(base: str | Path | None = None) -> dict[str, Any]:
    """Commit de git del código y si los archivos que determinan el resultado tenían
    cambios sin confirmar. Sin git disponible, ambos son None."""
    base = resolver_repo(base)
    try:
        commit = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        existentes = [r for r in RUTAS_CODIGO if (base / r).exists()]
        diff = subprocess.run(
            ["git", "-C", str(base), "diff", "--quiet", "HEAD", "--", *existentes], capture_output=True
        )
        modificado = None if diff.returncode not in (0, 1) else diff.returncode == 1
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "modificado": None}
    return {"commit": commit, "modificado": modificado}


def _relativa(ruta: Path, base: Path) -> str:
    return Path(ruta).resolve().relative_to(base.resolve()).as_posix()


def manifiesto_pedido(r: Rutas, nombre_modelo: str, version: str | int, ruta_modelo: str | Path) -> dict[str, Any]:
    """Qué produjo el pedido de `r`: entradas, modelo y código, con sus huellas."""
    base = r.base
    extra = {}
    if r.pronostico.exists():  # pronóstico diario por cuantil (desde la sección 11)
        extra["pronostico"] = {"ruta": _relativa(r.pronostico, base), "md5": md5_archivo(r.pronostico)}
    return {
        **extra,
        "clave": r.clave,
        "ds": r.ds,
        "conocimiento": r.conocimiento,
        "pedido": {"ruta": _relativa(r.pedido, base), "md5": md5_archivo(r.pedido)},
        "features": {"ruta": _relativa(r.features, base), "md5": md5_archivo(r.features)},
        "datos_crudos": {"ruta": _relativa(r.raw, base), "md5": md5_directorio(r.raw)},
        "modelo": {
            "nombre": nombre_modelo,
            "version": str(version),
            "ruta": _relativa(Path(ruta_modelo), base),
            "md5": md5_directorio(ruta_modelo),
        },
        # El commit se lee en la raíz del repo (en Docker, /opt/airflow/repo).
        "codigo": info_codigo(os.environ.get("INVENTARIO_REPO") or base),
    }


def guardar_manifiesto(manifiesto: dict[str, Any], ruta: str | Path) -> Path:
    """Escribe el manifiesto (JSON ordenado y estable: mismo contenido, mismo archivo)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ruta


def cargar_manifiesto(ruta: str | Path) -> dict[str, Any]:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


# --- DVC -----------------------------------------------------------------------------

Ejecutor = Callable[..., subprocess.CompletedProcess]


def _dvc(argumentos: list[str], base: Path, ejecutar: Ejecutor = subprocess.run) -> str:
    # `python -m dvc` con el mismo intérprete: no depende del PATH (contenedor, .venv, CI).
    resultado = ejecutar([sys.executable, "-m", "dvc", *argumentos], cwd=base, capture_output=True, text=True)
    if resultado.returncode != 0:
        raise RuntimeError(f"dvc {' '.join(argumentos)} falló: {(resultado.stderr or resultado.stdout).strip()}")
    return (resultado.stdout or "").strip()


def remoto_accesible(base: str | Path | None = None, ejecutar: Ejecutor = subprocess.run) -> tuple[bool, str]:
    """¿Se puede subir al remoto por defecto?

    Un remoto en disco debe existir y admitir escritura: si no está montado en el
    contenedor, `dvc push` lo crearía dentro del contenedor y los datos se perderían.
    Los remotos en la nube se dan por accesibles (las credenciales las valida `dvc push`).
    """
    base = resolver_repo(base)
    try:
        nombre = _dvc(["remote", "default"], base, ejecutar)
        url = _dvc(["config", f"remote.{nombre}.url"], base, ejecutar) if nombre else ""
    except RuntimeError:  # `dvc remote default` termina con error si no hay ninguno
        return False, "sin remoto por defecto"
    if not url:
        return False, "sin remoto por defecto"
    if "://" in url and not url.startswith("file://"):
        return True, f"remoto {nombre} ({url})"
    ruta = Path(url.removeprefix("file://"))
    if not ruta.is_absolute():
        ruta = (base / ".dvc" / ruta).resolve()
    if ruta.is_dir() and os.access(ruta, os.W_OK):
        return True, f"remoto {nombre} ({ruta})"
    return False, f"el remoto {nombre} ({ruta}) no existe o no admite escritura aquí"


@dataclass(frozen=True)
class ResultadoVersionado:
    rutas: tuple[str, ...]
    subido: bool
    detalle: str


def versionar_con_dvc(
    base: str | Path | None = None,
    rutas: tuple[str, ...] = RUTAS_VERSIONADAS,
    subir: bool = True,
    ejecutar: Ejecutor = subprocess.run,
) -> ResultadoVersionado:
    """`dvc add` de las rutas existentes y, si el remoto está accesible, `dvc push`.

    Sin cambios en los datos, `dvc add` deja los mismos punteros (idempotente). Un remoto
    no accesible (p. ej. un clon nuevo sin remoto montado) no es un error: los datos quedan
    en la caché local y se suben después con `dvc push`.
    """
    base = resolver_repo(base)
    existentes = tuple(r for r in rutas if (base / r).exists())
    if not existentes:
        return ResultadoVersionado((), False, "no hay nada que versionar")
    # --no-relink: DVC guarda en la caché sin volver a escribir el workspace desde ella.
    # Sin esto, borra y recrea los archivos ya rastreados y podría pisar los que un DAG
    # concurrente esté reescribiendo (p. ej. un reproceso de esa misma fecha).
    _dvc(["add", "-q", "--no-relink", *existentes], base, ejecutar)
    if not subir:
        return ResultadoVersionado(existentes, False, "solo caché local (subir=False)")
    accesible, detalle = remoto_accesible(base, ejecutar)
    if not accesible:
        log.warning("Datos versionados en la caché local de DVC, sin subir: %s", detalle)
        return ResultadoVersionado(existentes, False, detalle)
    _dvc(["push", "-q"], base, ejecutar)
    return ResultadoVersionado(existentes, True, detalle)
