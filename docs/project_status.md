# Estado del proyecto

Este documento resume el estado de `ulens_lsst` al inicio de su reorganización para publicación. No sustituye los registros detallados de procedencia ni demuestra validación científica.

## Resumen

| Área | Estado |
|---|---|
| Snapshot previo a refactorización | Completado |
| Backup privado verificado | Completado |
| Rama de desarrollo aislada | Completado |
| Procedencia DP0.2 reconstruida | Parcial |
| Validación de integridad DP0.2 | Pendiente |
| Validación fotométrica DP0.2 | Pendiente |
| Tests offline | Pendiente |
| Refactorización | No iniciada |
| DP1 end-to-end | Pendiente |
| Plan RAC | Pendiente |
| Release candidate | Pendiente |

<details>
<summary>Hechos verificados</summary>

- La reorganización comienza desde un worktree limpio basado en `origin/main`.
- El checkout previo y sus cambios locales permanecen separados.
- Existe un backup privado verificado fuera del repositorio.
- Hay productos de una campaña DP0.2 ejecutada, pero ejecución y validación científica son estados diferentes.

Véanse [el estado del repositorio](provenance/repository_state_2026-08-05.md) y [el estado de la campaña DP0.2](provenance/dp02_campaign_status.md).

</details>

<details>
<summary>Inferencias</summary>

- Una reorganización incremental, precedida por un inventario versionable, debería reducir el riesgo de perder procedencia. Esto es una inferencia de ingeniería, no un resultado científico.

</details>

<details>
<summary>Información desconocida</summary>

- El commit y el entorno exactos empleados para producir la campaña DP0.2.
- El grado de integridad, desempeño fotométrico y reproducibilidad científica de los productos.
- El estado end-to-end de DP1.

</details>

<details>
<summary>Decisiones pendientes</summary>

- Definir criterios de integridad y aceptación científica.
- Definir el plan RAC y el alcance de una futura release candidate.
- Diseñar tests offline, manifests de corrida y schemas canónicos.

La estrategia de ramas está registrada en [ENG-001](decisions/ENG-001-development-branch-strategy.md).

</details>

## Próximo hito

Crear un inventario estructurado y versionable de la campaña DP0.2, antes de modificar el comportamiento del pipeline.

