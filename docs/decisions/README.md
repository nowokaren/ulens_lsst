# Registros de decisiones

Este directorio conserva decisiones de ingeniería y científicas de forma explícita, revisable y trazable.

## Prefijos

- `ENG`: decisiones de ingeniería de software, infraestructura, packaging, interfaces o proceso de desarrollo.
- `SCI`: decisiones científicas sobre modelos, selecciones, métricas, supuestos, validación o interpretación.

El identificador debe ser secuencial dentro de cada prefijo, por ejemplo `ENG-001` o `SCI-001`.

## Estados permitidos

- `proposed`: propuesta abierta, todavía no adoptada.
- `accepted`: decisión adoptada y vigente.
- `superseded`: reemplazada por un registro posterior, que debe citarse.
- `rejected`: evaluada y descartada, conservando la justificación.

## Plantilla

```markdown
# <ID>: Título

- Estado: proposed | accepted | superseded | rejected
- Fecha: YYYY-MM-DD

## Contexto

## Pregunta

## Opciones consideradas

## Evidencia

## Experimento requerido

## Decisión

## Consecuencias

## Riesgos

## Criterio para revisar la decisión

## Referencias
```

## Hechos verificados

- Un registro aceptado documenta una decisión; no convierte sus supuestos en evidencia científica.
- La evidencia debe enlazar artefactos, resultados o fuentes revisables cuando estén disponibles.

## Inferencias

- Mantener separados los registros `ENG` y `SCI` debería facilitar la revisión por personas con responsabilidades diferentes.

## Información desconocida

- La política definitiva de revisión y aprobación para decisiones científicas.
- Quiénes serán responsables de aceptar o reemplazar cada clase de decisión.

## Decisiones pendientes

- Definir responsables y requisitos de aprobación para registros `SCI`.
- Definir cuándo un experimento debe quedar versionado junto con el registro.

Una propuesta generada por Codex no constituye evidencia científica. Puede servir como borrador, pregunta o alternativa, pero debe evaluarse contra evidencia científica independiente antes de aceptar una decisión `SCI`.

