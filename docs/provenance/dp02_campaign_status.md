# Estado de procedencia y validación de la campaña DP0.2

Este documento registra la evidencia disponible sobre la campaña `dataset_dp0_0` a `dataset_dp0_42`. Una finalización del CLI no se interpreta como validación científica.

## Hechos verificados

- Existen 43 directorios, desde `dataset_dp0_0` hasta `dataset_dp0_42`.
- Existen 43 configuraciones preservadas.
- Existen 46 logs `pipeline*.log`, debido a reintentos.
- Se solicitaron 107500 eventos.
- Hay 107500 resultados de procesamiento de eventos:
  - 107487 exitosos;
  - 13 fallidos.
- Hay 57425548 filas de fotometría ideal.
- Hay 50065706 filas de fotometría sobre calexps.
- Se evaluaron 70549 calexps:
  - 64535 exitosos;
  - 6014 fallidos.
- Los 43 runs contienen el marcador de finalización del CLI.
- El marcador de finalización no significa que todos los eventos y calexps hayan sido exitosos.
- Los 13 fallos de eventos están relacionados con la condición `DL >= DS`.
- El `batch_run.py` preservado actualmente ejecuta solamente índices específicos y no puede considerarse una copia demostrada del driver original de toda la campaña.
- No hay un SHA Git registrado en configs, logs o Parquet.
- Los tracebacks de campaña apuntan a un checkout diferente del repositorio actual.
- No se puede determinar con certeza el commit exacto utilizado.
- DP0.2 fue ejecutado end-to-end, pero no está científicamente validado.
- No se encontró evidencia inequívoca de que la etapa final de chi2 se haya producido para toda la campaña.

## Hipótesis de procedencia

- El código utilizado pudo estar relacionado con commits de septiembre u octubre de 2025, porque la cronología observada de commits y ejecuciones se superpone con ese período.
- También pudo contener cambios locales no registrados.

Estas son hipótesis. La evidencia disponible no permite asignar un commit a la campaña ni identificar un dirty state concreto. En particular, no hay evidencia para afirmar que `v1.0.0` haya sido la versión usada.

## Información desconocida

- El commit exacto utilizado.
- El dirty state original.
- El entorno exacto de ejecución.
- Las versiones exactas de las dependencias.
- La explicación de los 6014 calexps fallidos.
- La política científica aplicable a los 13 eventos con `DL >= DS`.
- La reproducibilidad exacta de las consultas TAP y Butler.
- El estado de la etapa chi2 para el conjunto completo.
- Los criterios científicos de aceptación de eventos, curvas y mediciones.

## Estado de validación

| Dimensión | Estado verificado | Alcance de la evidencia |
|---|---|---|
| Ejecución | Completada con fallos parciales | Hay productos, resultados y marcadores del CLI para 43 runs |
| Integridad | Pendiente | Hay conteos y checksums, pero falta un inventario semántico y validar relaciones entre tablas |
| Validación de inyección | Pendiente | No se identificó una prueba sistemática de verdad inyectada frente a verdad recuperada |
| Validación fotométrica | Pendiente | No se identificaron criterios documentados de sesgo, precisión o completitud |
| Calibración de incertidumbres | Pendiente | No se identificó una evaluación documentada de cobertura o calibración de errores |
| Detección | Pendiente | No se identificaron métricas de eficiencia, falsos positivos o función de selección |
| Reproducibilidad | Parcial | Existen configs, logs y checksums, pero faltan commit, entorno e identidad reproducible de consultas |

## Inferencias

- Los conteos permiten priorizar controles de integridad, pero no permiten inferir por sí solos calidad fotométrica o validez científica.
- Los reintentos y la ausencia de un SHA registrado aumentan el riesgo de que los 43 runs no compartan un único estado de código.

## Decisiones pendientes

- Definir el schema del inventario estructurado de campaña.
- Clasificar las causas de fallo de calexps y eventos.
- Definir criterios cuantitativos para cada dimensión de validación.
- Determinar si chi2 debe reconstruirse, reejecutarse o considerarse fuera del alcance de la campaña preservada.

