import os
import subprocess
import logging
from datetime import datetime

def setup_logging():
    """Configura el logging para el script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def run_command(command, cwd=None):
    """Ejecuta un comando git y retorna su salida"""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.stderr}"

def init_repo(repo_path):
    """Inicializa el repositorio si no existe"""
    if not os.path.exists(os.path.join(repo_path, '.git')):
        success, output = run_command(['git', 'init'], repo_path)
        if success:
            logging.info("Repositorio Git inicializado")
        else:
            logging.error(f"Error al inicializar repositorio: {output}")
            return False
    return True

def configure_remote(repo_path, remote_url):
    """Configura el repositorio remoto"""
    # Verificar si el remoto ya existe
    success, output = run_command(['git', 'remote', '-v'], repo_path)
    if success and 'origin' not in output:
        success, output = run_command(['git', 'remote', 'add', 'origin', remote_url], repo_path)
        if not success:
            logging.error(f"Error al configurar remoto: {output}")
            return False
        logging.info("Remoto configurado exitosamente")
    return True

def commit_and_push(repo_path, commit_message=None):
    """Realiza commit y push de los cambios"""
    # Agregar todos los cambios
    success, output = run_command(['git', 'add', '.'], repo_path)
    if not success:
        logging.error(f"Error al agregar cambios: {output}")
        return False

    # Crear mensaje de commit por defecto si no se proporciona uno
    if not commit_message:
        commit_message = f"Actualización automática: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # Realizar commit
    success, output = run_command(['git', 'commit', '-m', commit_message], repo_path)
    if not success:
        logging.error(f"Error al hacer commit: {output}")
        return False
    logging.info("Commit realizado exitosamente")

    # Push al repositorio remoto
    success, output = run_command(['git', 'push', '-u', 'origin', 'main'], repo_path)
    if not success:
        # Intentar con master si main falla
        success, output = run_command(['git', 'push', '-u', 'origin', 'master'], repo_path)
        if not success:
            logging.error(f"Error al hacer push: {output}")
            return False
    logging.info("Push realizado exitosamente")
    return True

def main():
    """Función principal"""
    setup_logging()
    
    # Configuración del repositorio
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    remote_url = "https://github.com/Ogironr/NN_Insuficiencia_cardiaca_Kaggle.git"  # Reemplazar con tu URL
    
    # Inicializar repositorio
    if not init_repo(repo_path):
        return
    
    # Configurar remoto
    if not configure_remote(repo_path, remote_url):
        return
    
    # Realizar commit y push
    commit_message = "Actualización del pipeline MLOps"
    if commit_and_push(repo_path, commit_message):
        logging.info("Proceso de git completado exitosamente")
    else:
        logging.error("Error en el proceso de git")

if __name__ == "__main__":
    main()
