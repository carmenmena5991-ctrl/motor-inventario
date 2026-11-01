"""Reproduce un pedido publicado a partir de su versión de código y de datos.

Uso (desde la raíz del repo, en Ubuntu):
    git checkout <commit con los punteros .dvc de esa publicación>
    dvc checkout                       # o dvc pull, si los datos no están en la caché local
    python scripts/reproducir_pedido.py --clave 2017-07-31

Sale con código 0 si el pedido regenerado es idéntico (byte a byte) al publicado.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from inventario.reproduccion import reproducir_pedido  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clave", required=True, help="Fecha del pedido (o <ds>__conocimiento_<fecha>)")
    parser.add_argument("--base", type=Path, default=RAIZ, help="Raíz del proyecto (por defecto, este repo)")
    args = parser.parse_args()

    r = reproducir_pedido(args.clave, args.base)
    print(f"Pedido {r.clave}: modelo v{r.modelo_version}, código {r.commit or 'desconocido'}")
    for nombre, ok in r.verificaciones.items():
        print(f"  {nombre:<13} {'coincide' if ok else 'NO coincide'} con el manifiesto")
    print(f"  md5 publicado   {r.md5_publicado or '(no existe)'}")
    print(f"  md5 reproducido {r.md5_reproducido or '(no se regeneró)'}  ({r.filas} filas)")
    print("Resultado: IDÉNTICO" if r.identico else "Resultado: DISTINTO")
    return 0 if r.identico else 1


if __name__ == "__main__":
    raise SystemExit(main())
