# Estado del repositorio al 2026-08-05

Este registro documenta el punto de partida de la reorganización. No publica el contenido ni la ruta absoluta del backup privado.

## Hechos verificados

- Rama base: `origin/main`.
- Commit base: `cfbd635938e857903dee2b1315cb93b16d53e775`.
- Rama de desarrollo: `refactor/publication-ready`.
- El nuevo worktree comenzó limpio.
- El checkout original conserva dos modificaciones locales:
  - `docs/tutorials/tutorial_RSP.ipynb`;
  - `ulens_lsst/simulation_pipeline.py`.
- Esas modificaciones no fueron aplicadas a la rama nueva.
- El patch local de `simulation_pipeline.py` contiene una implementación parcial de `num_cores` y un cambio accidental en un docstring.
- El notebook local contiene outputs, un traceback y una URL S3 firmada histórica. Por ese motivo, su copia binaria exacta fue preservada solamente en el backup privado y no fue incorporada a esta rama.
- El backup contiene 110 archivos verificados mediante SHA256.
- Se calcularon checksums para 458 archivos científicos sin copiar sus contenidos al backup.
- El backup ocupa aproximadamente 151 MB.
- Existen dos worktrees separados: el checkout original y el worktree `publication-ready`.
- Durante el procedimiento no se hizo `fetch`, commit ni push.

## Limitación del procedimiento de backup

El primer intento de generar los checksums de productos científicos utilizó un directorio base incorrecto. El procedimiento se detuvo antes de crear el worktree. El archivo de checksums se corrigió dentro del backup parcial y la verificación final terminó correctamente. No se eliminó ni se reemplazó silenciosamente el backup parcial.

## Entorno observado

La siguiente información describe el entorno observado el 2026-08-05. No representa necesariamente el entorno utilizado para producir la campaña DP0.2 de 2025.

### Hechos verificados

- EUPS estaba disponible.
- `lsst_distrib` no estaba configurado en esa shell.
- `numpy`, `pyarrow`, `rubin_sim` y `lsst.rsp` eran importables.
- `pyLIMA`, `lsst.daf.butler`, `lsst.afw`, `lsst.meas` y `lsst.source.injection` no eran importables en esa shell.

No se diagnostica aquí la causa de esas condiciones.

## Inferencias

- La separación entre worktrees reduce el riesgo de mezclar accidentalmente los cambios locales previos con la reorganización. Esto es una inferencia operativa.
- El backup aporta una referencia previa al refactor, pero no demuestra por sí solo la procedencia completa de productos científicos de 2025.

## Información desconocida

- Si el checkout original conserva toda la información necesaria para reproducir la campaña DP0.2.
- El entorno exacto utilizado para la campaña de 2025.
- La relación exacta entre los cambios locales preservados y ejecuciones históricas.

## Decisiones pendientes

- Definir durante cuánto tiempo y bajo qué política se conservará el backup privado.
- Definir qué metadatos del entorno deberán integrar futuros manifests de corrida.
- Decidir si algún cambio local previo será rediseñado e incorporado posteriormente mediante commits independientes.

