#!/bin/bash
# backup.sh — Script de Backup Diario
# Uso: bash backup.sh <IP_DEL_DROPLET>
# Ejemplo: bash backup.sh 123.456.789.0

IP=$1
# Auto-detectar clave SSH disponible (ed25519 tiene prioridad sobre rsa)
if [ -n "$AMD_SSH_KEY" ]; then
    SSH_KEY="$AMD_SSH_KEY"
elif [ -f "$HOME/.ssh/id_ed25519" ]; then
    SSH_KEY="$HOME/.ssh/id_ed25519"
elif [ -f "$HOME/.ssh/id_rsa" ]; then
    SSH_KEY="$HOME/.ssh/id_rsa"
else
    echo "ERROR: No se encontró ninguna clave SSH en ~/.ssh/"
    echo "Especifica la clave con: AMD_SSH_KEY=/ruta/a/clave bash backup.sh <IP>"
    exit 1
fi
FECHA=$(date +%Y-%m-%d)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_CHECKPOINTS="$SCRIPT_DIR/checkpoints/backup_$FECHA"

if [ -z "$IP" ]; then
    echo "ERROR: Debes proporcionar la IP del Droplet."
    echo "Uso: bash backup.sh <IP>"
    exit 1
fi

echo "==========================================================="
echo " BACKUP DIARIO — AMD MI300X"
echo " Droplet IP : $IP"
echo " Fecha      : $FECHA"
echo "==========================================================="

# ── Paso 1: Git commit + push desde el Droplet ───────────────────
echo "[1/3] Haciendo git commit y push desde el Droplet..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no root@$IP << 'REMOTE'
    cd /root/workspace
    FECHA=$(date +%Y-%m-%d)

    # Configurar identidad git si no está configurada
    git config user.email "guillermoveliz231@hotmail.com" 2>/dev/null || true
    git config user.name "AMD Hackathon" 2>/dev/null || true

    # Verificar que hay un remote configurado
    if git remote -v 2>/dev/null | grep -q origin; then
        git add -A
        git diff --cached --quiet && echo "  Sin cambios nuevos." || \
            (git commit -m "Backup diario $FECHA [auto]" && git push && echo "  Push exitoso.")
    else
        echo "  ADVERTENCIA: No hay remote git configurado. Omitiendo push."
        echo "  Crea el repo y ejecuta: git remote add origin <URL>"
    fi
REMOTE

# ── Paso 2: Copiar pesos del modelo (.pt) a local ────────────────
echo "[2/3] Copiando pesos del modelo (.pt) al PC local..."
mkdir -p "$LOCAL_CHECKPOINTS"

# Intentar copiar desde el host (volumen montado) o desde el contenedor
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "root@$IP:/root/workspace/checkpoints/*.pt" \
    "$LOCAL_CHECKPOINTS/" 2>/dev/null \
    && echo "  Pesos copiados en: $LOCAL_CHECKPOINTS/" \
    || echo "  Sin archivos .pt aún (normal en Días 1-2)."

# ── Paso 3: Copiar logs relevantes ───────────────────────────────
echo "[3/3] Copiando logs de entrenamiento..."
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "root@$IP:/root/workspace/logs/*" \
    "$LOCAL_CHECKPOINTS/" 2>/dev/null \
    && echo "  Logs copiados." \
    || echo "  Sin logs aún."

echo ""
echo "==========================================================="
echo " BACKUP COMPLETADO — $FECHA"
echo "==========================================================="
echo " Archivos locales en: $LOCAL_CHECKPOINTS/"
ls "$LOCAL_CHECKPOINTS/" 2>/dev/null && echo "" || echo " (directorio vacío)"
echo ""
echo " PRÓXIMO PASO MANUAL:"
echo "   Destruye el Droplet en: https://cloud.digitalocean.com"
echo "   Panel > GPU Droplets > [tu droplet] > Destroy"
echo "==========================================================="
