# Mapa del repositorio para el anexo

## Propósito

El repositorio permite:

1. auditar los 18 registros utilizados en el paper sin GPU;
2. regenerar tablas y figuras a partir de esos registros;
3. repetir el procedimiento físico como una corrida nueva.

## Flujo

```text
escenario + base sintética + configuración congelada
                         ↓
             infera_session_runner.py
                         ↓
       18 JSONL + 18 manifiestos + logs
                         ↓
              analiza_tres_brazos.py
                         ↓
       tablas + informe + manifiesto de análisis
                         ↓
              figuras_tres_brazos.py
                         ↓
           PNG + PDF + manifiesto de figuras
```

La medición física ocurre en el runner. Analizador y generador de figuras
trabajan sobre registros persistidos; no inventan ni simulan energía.

## Componentes públicos

- `experimentos/experimento_principal/evidencia/`: registros originales y
  resultados derivados.
- `experimentos/experimento_principal/paquete_preservado/`: contenedor exacto
  y checksum externo.
- `experimentos/experimento_principal/figuras/`: derivados visuales
  regenerables.
- `infera/`: instrumentación, políticas, análisis, auditor y pruebas.
- `infera/config/reference/`: escenario y configuración de referencia.
- `infera/kb/`: base sintética utilizada.
- `docs/`: instalación, reproducción, configuración, salidas y límites.

## Diseño publicado

```text
3 políticas × 2 representaciones × 3 repeticiones instrumentales
= 18 sesiones de 29 tareas
```

Los nombres `completo`, `resumen` y `descarte` aparecen en cada JSONL. `rep1`,
`rep2` y `rep3` describen pasadas instrumentales, no casos de calidad
independientes.

## Comando de auditoría

Desde la raíz:

```bash
python3 infera/audita_paquete_tres_brazos.py \
  --campaign experimentos/experimento_principal/evidencia \
  --archive experimentos/experimento_principal/paquete_preservado/experimento_principal_evidencia.tar.gz.bin \
  --checksum experimentos/experimento_principal/paquete_preservado/experimento_principal_evidencia.tar.gz.sha256 \
  --reanalysis /tmp/infera_reanalysis
```

El resultado esperado declara 18 sesiones, 29 tareas, cero parciales,
integridad válida y reanálisis idéntica.

## Controles de una corrida nueva

- configuración explícita y salida protegida contra sobrescritura;
- tareas, base, tokenizadores, software y hashes congelados;
- presupuesto real contado con plantilla de chat;
- NVML obligatorio y energía positiva;
- proceso GPU exclusivo y telemetría completa;
- respuestas, razones de cierre, tiempos, tokens y trazas persistidos;
- orden contrabalanceado;
- manifiestos por sesión y por experimento;
- publicación solo con 18 sesiones completas.

La numeración y el nombre del anexo deben corresponder con la versión final del
manuscrito.
