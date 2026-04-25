#!/bin/bash
# setup_env.sh — Script de automatización diaria de infraestructura AMD MI300X
# Ejecutar en el Droplet recién creado: bash setup_env.sh

set -e  # Detener si cualquier comando falla

echo "==========================================================="
echo " Iniciando configuración del entorno — Agente Clínico AMD"
echo "==========================================================="

# ── 1. Sistema anfitrión ─────────────────────────────────────────
echo "[1/5] Actualizando sistema e instalando dependencias base..."
sudo apt-get update -qq
sudo apt-get install -y git tmux htop wget curl build-essential rsync

# ── 2. Espacio de trabajo ────────────────────────────────────────
echo "[2/5] Creando estructura de directorios..."
mkdir -p /root/workspace/{data/mimic3,checkpoints,logs}
cd /root/workspace

# Clonar repositorio si existe y no está ya descargado
if [ ! -d ".git" ]; then
    echo "  Clonando repositorio..."
    git clone https://github.com/Guivve-A/INDEPENDENT-CLINICAL-RECONSTRUCTION-AGENT- . \
        && echo "  Repositorio clonado." \
        || echo "  Repositorio no disponible aún — continuando sin clonar."
else
    echo "  Repositorio ya presente — haciendo pull..."
    git pull || echo "  Pull falló (posible repo vacío) — continuando."
fi

# ── 3. Directorios MIMIC-III ─────────────────────────────────────
echo "[3/5] Preparando directorios para datos MIMIC-III..."
mkdir -p /root/workspace/data/mimic3/{raw,processed}
echo "  Directorios listos. Agente 1 añadirá los comandos wget de PhysioNet."

# ── 4. Docker ────────────────────────────────────────────────────
echo "[4/5] Verificando Docker..."
if ! command -v docker &> /dev/null; then
    echo "  Docker no detectado — instalando..."
    sudo apt-get install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    usermod -aG docker root
    echo "  Docker instalado."
else
    echo "  Docker ya disponible: $(docker --version)"
fi

# Detener contenedor anterior si existe (rearranque diario limpio)
if sudo docker ps -a --format '{{.Names}}' | grep -q '^entorno_agente$'; then
    echo "  Deteniendo contenedor anterior..."
    sudo docker stop entorno_agente && sudo docker rm entorno_agente
fi

# ── 5. Contenedor ROCm ───────────────────────────────────────────
echo "[5/5] Levantando contenedor rocm/primus:v26.2..."
sudo docker run -d \
    --name entorno_agente \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add=video \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v /root/workspace:/workspace \
    rocm/primus:v26.2 sleep infinity

echo ""
echo "==========================================================="
echo " ENTORNO LISTO"
echo "==========================================================="
echo " Contenedor activo: $(sudo docker ps --filter name=entorno_agente --format '{{.Status}}')"
echo ""
echo " Comandos útiles:"
echo "   Entrar al contenedor : sudo docker exec -it entorno_agente /bin/bash"
echo "   Ver logs GPU         : sudo docker exec entorno_agente rocminfo | grep 'Agent Type'"
echo "   Ver uso VRAM         : sudo docker exec entorno_agente rocm-smi"
echo "==========================================================="
