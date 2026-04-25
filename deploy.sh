#!/bin/bash
# deploy.sh — Script de Despliegue Diario
# Uso: bash deploy.sh <IP_DEL_DROPLET>
# Ejemplo: bash deploy.sh 123.456.789.0

set -e

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
    echo "Especifica la clave con: AMD_SSH_KEY=/ruta/a/clave bash deploy.sh <IP>"
    exit 1
fi

if [ -z "$IP" ]; then
    echo "ERROR: Debes proporcionar la IP del Droplet."
    echo "Uso: bash deploy.sh <IP>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==========================================================="
echo " DESPLIEGUE — AMD MI300X"
echo " Droplet IP : $IP"
echo " SSH Key    : $SSH_KEY"
echo "==========================================================="

# ── Paso 1: Verificar conectividad ───────────────────────────────
echo "[1/3] Verificando conexión SSH con el Droplet..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    root@$IP "echo '  Conexión OK'"

# ── Paso 2: Subir el script de configuración ─────────────────────
echo "[2/3] Subiendo setup_env.sh al Droplet..."
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "$SCRIPT_DIR/setup_env.sh" root@$IP:/root/setup_env.sh
echo "  Archivo subido."

# ── Paso 3: Ejecutar el script en el Droplet ─────────────────────
echo "[3/3] Ejecutando setup_env.sh en el Droplet (puede tomar ~3 min)..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no root@$IP \
    "chmod +x /root/setup_env.sh && bash /root/setup_env.sh"

echo ""
echo "==========================================================="
echo " DESPLIEGUE COMPLETADO"
echo "==========================================================="
echo " Para entrar al servidor:"
echo "   ssh -i $SSH_KEY root@$IP"
echo ""
echo " Para entrar al contenedor GPU:"
echo "   sudo docker exec -it entorno_agente /bin/bash"
echo "==========================================================="
