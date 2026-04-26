#!/bin/bash
# setup_env.sh — Configuración del entorno en AMD Developer Cloud
# Ejecutar una vez por instancia: bash setup_env.sh
# Sin Docker — PyTorch y ROCm disponibles directamente en el host

set -e

echo "==========================================================="
echo " Configuración del entorno — Agente Clínico AMD"
echo " AMD MI300X | ROCm 7.0.0 | PyTorch 2.9.0.dev+rocm7.0.0"
echo "==========================================================="

# ── 1. Dependencias del sistema ────────────────────────────────────
echo "[1/4] Instalando dependencias del sistema..."
apt-get update -qq
apt-get install -y git python3-venv python3-pip wget curl htop tmux \
    build-essential rsync python3-dev

# ── 2. Workspace y repositorio ─────────────────────────────────────
echo "[2/4] Preparando workspace en ~/workspace..."
mkdir -p ~/workspace/{data/mimic3/raw,data/mimic3/processed,checkpoints,logs}
cd ~/workspace

if [ ! -d ".git" ]; then
    echo "  Clonando repositorio..."
    git clone https://github.com/Guivve-A/INDEPENDENT-CLINICAL-RECONSTRUCTION-AGENT- . \
        && echo "  Repositorio clonado." \
        || echo "  Repositorio no disponible aún — continuando sin clonar."
else
    echo "  Repositorio ya presente — haciendo pull..."
    git pull || echo "  Pull falló (repo vacío) — continuando."
fi

# Configurar identidad git
git config user.email "guillermoveliz231@hotmail.com" 2>/dev/null || true
git config user.name "AMD Hackathon" 2>/dev/null || true

# ── 3. Entorno virtual Python ──────────────────────────────────────
echo "[3/4] Creando/actualizando entorno virtual ~/dispositivo_ia..."
if [ ! -d ~/dispositivo_ia ]; then
    python3 -m venv ~/dispositivo_ia --system-site-packages
    echo "  Venv creado (con acceso a paquetes del sistema — incluye PyTorch ROCm)"
else
    echo "  Venv ya existe — omitiendo creación."
fi

source ~/dispositivo_ia/bin/activate

# Instalar dependencias del proyecto (PyTorch viene del sistema ROCm)
pip install --upgrade pip -q
if [ -f ~/workspace/requirements.txt ]; then
    echo "  Instalando requirements.txt..."
    pip install -r ~/workspace/requirements.txt -q
else
    echo "  requirements.txt no encontrado — instalando dependencias base..."
    pip install wfdb scipy numpy fastapi uvicorn websockets -q
fi

# ── 4. Verificar GPU y PyTorch ─────────────────────────────────────
echo "[4/4] Verificando hardware GPU y PyTorch..."
python3 -c "
import torch
print(f'  PyTorch   : {torch.__version__}')
print(f'  CUDA/ROCm : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU       : {torch.cuda.get_device_name(0)}')
    free, total = torch.cuda.mem_get_info()
    print(f'  VRAM      : {total/1e9:.1f} GB total, {free/1e9:.1f} GB libre')
else:
    print('  ADVERTENCIA: GPU no detectada — verificar ROCm')
"

echo ""
echo "==========================================================="
echo " ENTORNO LISTO"
echo "==========================================================="
echo " Workspace  : ~/workspace"
echo " Venv       : ~/dispositivo_ia  (--system-site-packages)"
echo ""
echo " Para trabajar cada sesión:"
echo "   source ~/dispositivo_ia/bin/activate"
echo "   cd ~/workspace"
echo ""
echo " Verificar GPU:"
echo "   python3 -c \"import torch; print(torch.cuda.get_device_name(0))\""
echo ""
echo " Backup nocturno (antes de destruir la instancia):"
echo "   cd ~/workspace && git branch -M main && git add -A && \\"
echo "   git commit -m \"Backup \$(date +%Y-%m-%d)\" --allow-empty && \\"
echo "   git push origin main"
echo ""
echo " Puertos UFW para Día 3+:"
echo "   sudo ufw allow 8000 && sudo ufw allow 3000"
echo "==========================================================="
