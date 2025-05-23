# Guía de Uso del Contenedor Docker para MLOps

## Prerequisitos
- Docker Desktop instalado
- Imagen base `nemo-gpu-dev` disponible localmente
- Soporte para GPU (drivers NVIDIA instalados)

## Verificar Imágenes Disponibles
```bash
docker images
```
Deberías ver la imagen `nemo-gpu-dev` en la lista.

## Crear un Nuevo Contenedor
Para crear un nuevo contenedor basado en la imagen existente:
```bash
docker run --name windsurf-mlops -it --gpus all -v "//c/Users/Giron/_ogironr/Proyectos/MLOps/Trabajo_Parcial:/workspace" nemo-gpu-dev bash
```

Donde:
- `--name windsurf-mlops`: Nombre del contenedor
- `-it`: Modo interactivo con terminal
- `--gpus all`: Habilita acceso a todas las GPUs
- `-v "ruta_local:/workspace"`: Monta el directorio del proyecto en el contenedor
- `nemo-gpu-dev`: Imagen base a usar
- `bash`: Comando a ejecutar al iniciar

## Gestión del Contenedor

### Ver Contenedores Existentes
```bash
docker ps -a
```

### Iniciar un Contenedor Existente
```bash
docker start windsurf-mlops
```

### Acceder al Contenedor en Ejecución
```bash
docker exec -it windsurf-mlops bash
```

### Detener el Contenedor
```bash
docker stop windsurf-mlops
```

### Sincronización de Fecha del Sistema
El contenedor Docker puede tener una fecha del sistema diferente a la del host, lo que puede causar problemas con los timestamps en los archivos generados. Si encuentras problemas relacionados con fechas (por ejemplo, fechas futuras en los logs), puedes sincronizar la fecha del contenedor con el host:

```bash
# Desde el host, obtén la fecha actual en formato ISO
docker exec windsurf-mlops date -s "$(date -Iseconds)"
```

### Eliminar el Contenedor
Si necesitas eliminar el contenedor para crearlo de nuevo:
```bash
docker rm windsurf-mlops
```

## Verificación del Montaje
Una vez dentro del contenedor, verifica que los archivos del proyecto estén disponibles:
```bash
ls /workspace
```
Deberías ver todos los archivos de tu proyecto local.

## Flujo de Trabajo Típico

1. **Primera Vez**:
   ```bash
   docker run --name windsurf-mlops -it --gpus all -v "//c/Users/Giron/_ogironr/Proyectos/MLOps/Trabajo_Parcial:/workspace" nemo-gpu-dev bash
   ```

2. **Sesiones Posteriores**:
   ```bash
   # Iniciar el contenedor
   docker start windsurf-mlops
   
   # Acceder al contenedor
   docker exec -it windsurf-mlops bash
   ```

3. **Al Terminar**:
   ```bash
   # Salir del contenedor (desde dentro del contenedor)
   exit
   
   # Detener el contenedor (opcional)
   docker stop windsurf-mlops
   ```

## Notas Importantes
- Los cambios en los archivos del directorio montado (`/workspace`) se reflejan inmediatamente tanto dentro como fuera del contenedor
- El contenedor mantiene su estado entre sesiones, incluyendo cualquier instalación adicional de paquetes
- Para ver los contenedores detenidos en Docker Desktop, desactiva el filtro "Only show running containers"
