import { AppProps } from 'next/app';
import { Toaster } from 'react-hot-toast';
import '../styles/globals.css';

// embracingearth.space - Premium DDSP Neural Cello App
export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Component {...pageProps} />
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#1e293b',
            color: '#fff',
            border: '1px solid #7c3aed',
          },
        }}
      />
    </>
  );
}





