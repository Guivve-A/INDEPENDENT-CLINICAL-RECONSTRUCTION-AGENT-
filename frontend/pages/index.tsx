/**
 * pages/index.tsx — Punto de entrada del monitor ECG.
 *
 * ECGDashboard usa WebGL (webgl-plot) que es exclusivamente browser.
 * Se carga con `ssr: false` para evitar que Next.js intente renderizarlo
 * en el servidor (Node.js no tiene WebGL context → crash en SSR).
 *
 * Todo el código de WebGL + WebSocket vive en components/ECGDashboard.tsx.
 */

import type { NextPage } from 'next';
import dynamic from 'next/dynamic';

// Carga el dashboard solo en el cliente — sin SSR
const ECGDashboard = dynamic(
  () => import('../components/ECGDashboard'),
  {
    ssr: false,
    loading: () => (
      <div
        style={{
          background: '#080c08',
          height: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#00FF88',
          fontFamily: 'monospace',
          fontSize: 13,
          letterSpacing: 2,
        }}
      >
        INICIALIZANDO MONITOR ECG...
      </div>
    ),
  }
);

const Home: NextPage = () => <ECGDashboard />;

export default Home;
