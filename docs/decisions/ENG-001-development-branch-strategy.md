# ENG-001: Estrategia de ramas durante la reorganización

- Estado: accepted
- Fecha: 2026-08-05

## Contexto

El checkout previo contiene cambios locales que deben preservarse, pero no incorporarse automáticamente. La reorganización necesita un punto de partida limpio y trazable antes de modificar el comportamiento del pipeline.

## Pregunta

¿Cómo debe aislarse y organizarse el desarrollo para conservar el estado previo, permitir revisión incremental y evitar mezclar refactorización con cambios científicos no validados?

## Opciones consideradas

1. Continuar directamente sobre `main` en el checkout original.
2. Crear una rama nueva en el checkout original y aplicar los cambios locales existentes.
3. Mantener `main` sin cambios y usar una rama y un worktree separados, sin aplicar automáticamente los cambios locales previos.

## Evidencia

### Hechos verificados

- El checkout original conserva dos modificaciones locales.
- Se creó un worktree limpio desde `origin/main` en el commit `cfbd635938e857903dee2b1315cb93b16d53e775`.
- El checkout original y el nuevo worktree permanecen separados.
- Los cambios locales previos no se aplicaron a la nueva rama.
- Existe un backup privado verificado previo a la reorganización.

## Experimento requerido

No se requiere un experimento científico para adoptar esta estrategia de ingeniería. Antes de integrar cambios que alteren resultados sí se requerirán tests y revisión adecuados al riesgo.

## Decisión

- Mantener `main` sin cambios durante la reorganización.
- Trabajar en `refactor/publication-ready`.
- Conservar separado el checkout original.
- No aplicar automáticamente los cambios locales previos.
- Usar commits pequeños y temáticos.
- Crear ramas temporales para tareas experimentales cuando sea necesario.
- No crear todavía una nueva release ni un nuevo tag.
- No mover `v1.0.0`.
- Integrar cambios solamente después de tests y revisión.
- Evitar commits grandes y ambiguos del tipo `reorganize project`.

## Orden inicial esperado de commits

1. Documentación de procedencia.
2. Inventario estructurado de campaña.
3. Tests de caracterización.
4. Configuración validada.
5. Manifest de corrida.
6. Schemas canónicos.
7. Adapters de colecciones.
8. Migración DP0.2.
9. Validación DP1.

## Consecuencias

- Los cambios serán más fáciles de revisar, revertir y relacionar con evidencia.
- Los cambios locales históricos deberán reevaluarse y, si corresponde, rediseñarse como trabajo nuevo.
- La integración requerirá disciplina para mantener separados cambios documentales, de infraestructura, de comportamiento y científicos.

## Riesgos

- La secuencia puede aumentar el número de commits y ramas temporales.
- Un cambio que mezcle categorías puede debilitar la trazabilidad.
- Mantener dos worktrees puede causar confusión si no se verifica siempre la ruta y rama activas.

## Inferencias

- Los commits pequeños deberían facilitar la atribución causal cuando cambien tests o resultados. Esta expectativa es una inferencia de ingeniería.

## Información desconocida

- Qué cambios locales previos resultarán útiles después del rediseño.
- Cuántas ramas experimentales serán necesarias.
- Qué criterios concretos habilitarán una futura release candidate.

## Decisiones pendientes

- Política de merge y revisión.
- Convención para ramas experimentales.
- Requisitos de CI y validación antes de integrar cada etapa.
- Criterios para crear una release candidate y un nuevo tag.

## Criterio para revisar la decisión

Revisar esta estrategia si el aislamiento por worktrees impide pruebas reproducibles, si el orden de commits bloquea dependencias necesarias o cuando exista evidencia suficiente para definir el proceso de release.

## Referencias

- [Estado del proyecto](../project_status.md)
- [Estado del repositorio](../provenance/repository_state_2026-08-05.md)
- [Estado de la campaña DP0.2](../provenance/dp02_campaign_status.md)

