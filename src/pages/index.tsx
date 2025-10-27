import React from 'react';
import Head from 'next/head';

// embracingearth.space - Premium DDSP Neural Cello Interface
const DDSPNeuralCello = () => {
  // For now, we'll use the static HTML file
  // The beautiful index_fixed.html is ready to use with proper styling
  return (
    <>
      <Head>
        <title>DDSP Neural Cello - embracingearth.space</title>
        <meta name="description" content="Premium AI Audio Synthesis - DDSP Neural Cello" />
      </Head>
      <div style={{ margin: '20px', textAlign: 'center', color: 'white' }}>
        <h1 style={{ fontSize: '24px', marginBottom: '20px' }}>DDSP Neural Cello</h1>
        <p style={{ marginBottom: '20px' }}>embracingearth.space - Premium AI Audio Synthesis</p>
        <p>Please use the beautiful static HTML interface at:</p>
        <a href="/index_fixed.html" style={{ color: '#9333ea', fontSize: '18px', textDecoration: 'underline' }}>
          Open Beautiful Interface
        </a>
        <div style={{ marginTop: '20px', padding: '20px', background: 'rgba(0,0,0,0.3)', borderRadius: '8px' }}>
          <p style={{ marginBottom: '10px' }}>Backend API: <a href="http://localhost:8000" style={{ color: '#3b82f6' }}>http://localhost:8000</a></p>
          <p style={{ marginBottom: '10px' }}>Upload endpoint: POST http://localhost:8000/api/upload-midi</p>
          <p>Download endpoint: GET http://localhost:8000/api/download/{'{filename}'}</p>
        </div>
      </div>
    </>
  )
}

export default DDSPNeuralCello;

  // Audio quality presets - embracingearth.space standards
  const qualityPresets = {
    draft: { label: 'Draft', description: 'Fast processing, lower quality', color: 'bg-gray-500' },
    standard: { label: 'Standard', description: 'Balanced quality/speed', color: 'bg-blue-500' },
    professional: { label: 'Professional', description: 'High quality, slower', color: 'bg-purple-500' },
    mastering: { label: 'Mastering', description: 'Maximum quality, slowest', color: 'bg-yellow-500' }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.mid') && !file.name.toLowerCase().endsWith('.midi')) {
      toast.error('Please upload a MIDI file (.mid or .midi)');
      return;
    }

    setCurrentFile(file);
    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('http://localhost:8000/api/upload-midi', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          toast.loading(`Uploading... ${progress}%`, { id: 'upload' });
        }
      });

      toast.success('MIDI uploaded successfully!', { id: 'upload' });
      
      setGeneratedAudio({
        url: `http://localhost:8000/api/download/${response.data.output_file.split('/').pop()}`,
        filename: response.data.original_filename,
        duration: response.data.duration,
        quality: response.data.quality_level,
        format: response.data.format,
        bitDepth: response.data.bit_depth,
        mastering: response.data.mastering_applied
      });

    } catch (error) {
      toast.error('Upload failed: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsUploading(false);
    }
  };

  const handlePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleDownload = () => {
    if (generatedAudio) {
      const link = document.createElement('a');
      link.href = generatedAudio.url;
      link.download = `cello_synthesis_${Date.now()}.${generatedAudio.format}`;
      link.click();
    }
  };

  const startTraining = async () => {
    try {
      const response = await axios.post('http://localhost:8000/api/training/start');
      toast.success('Training started!');
      
      // Poll training status
      const pollStatus = setInterval(async () => {
        try {
          const statusResponse = await axios.get('http://localhost:8000/api/training/status');
          setTrainingStatus(statusResponse.data);
          
          if (statusResponse.data.status === 'completed') {
            clearInterval(pollStatus);
            toast.success('Model training completed!');
          }
        } catch (error) {
          console.error('Status polling error:', error);
        }
      }, 2000);
      
    } catch (error) {
      toast.error('Failed to start training: ' + (error.response?.data?.detail || error.message));
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <motion.header 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-black/20 backdrop-blur-md border-b border-purple-500/20"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-purple-600 rounded-lg">
                <Music className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">DDSP Neural Cello</h1>
                <p className="text-purple-300 text-sm">embracingearth.space - Premium AI Audio Synthesis</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-2 bg-purple-600/20 hover:bg-purple-600/40 rounded-lg transition-colors"
              >
                <Settings className="h-5 w-5 text-purple-300" />
              </button>
              
              <button
                onClick={startTraining}
                disabled={trainingStatus?.status === 'loading' || trainingStatus?.status === 'processing'}
                className="px-4 py-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white font-medium transition-all"
              >
                <Zap className="h-4 w-4 inline mr-2" />
                Train Model
              </button>
            </div>
          </div>
        </div>
      </motion.header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Upload Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-dark rounded-2xl p-6"
          >
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Upload className="h-5 w-5 mr-2 text-purple-400" />
              Upload MIDI File
            </h2>
            
            <div className="space-y-4">
              {/* File Upload Area */}
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-purple-500/30 rounded-lg p-8 text-center cursor-pointer hover:border-purple-500/50 transition-colors"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".mid,.midi"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                
                <div className="space-y-3">
                  <div className="mx-auto w-12 h-12 bg-purple-600/20 rounded-full flex items-center justify-center">
                    <Upload className="h-6 w-6 text-purple-400" />
                  </div>
                  
                  <div>
                    <p className="text-white font-medium">Drop your MIDI file here</p>
                    <p className="text-purple-300 text-sm">or click to browse</p>
                  </div>
                  
                  <p className="text-gray-400 text-xs">
                    Supports .mid and .midi files
                  </p>
                </div>
              </div>

              {/* Current File Info */}
              {currentFile && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="bg-purple-600/10 rounded-lg p-4 border border-purple-500/20"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-white font-medium">{currentFile.name}</p>
                      <p className="text-purple-300 text-sm">
                        {(currentFile.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                    <div className="text-purple-400">
                      <Music className="h-5 w-5" />
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Quality Settings */}
              <div className="space-y-3">
                <label className="text-white font-medium">Audio Quality</label>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(qualityPresets).map(([key, preset]) => (
                    <button
                      key={key}
                      onClick={() => setAudioQuality(key)}
                      className={`p-3 rounded-lg border transition-all ${
                        audioQuality === key
                          ? 'border-purple-500 bg-purple-600/20'
                          : 'border-gray-600 bg-gray-800/20 hover:border-gray-500'
                      }`}
                    >
                      <div className="text-left">
                        <div className="flex items-center space-x-2">
                          <div className={`w-2 h-2 rounded-full ${preset.color}`} />
                          <span className="text-white text-sm font-medium">{preset.label}</span>
                        </div>
                        <p className="text-gray-400 text-xs mt-1">{preset.description}</p>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>

          {/* Generated Audio Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-dark rounded-2xl p-6"
          >
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Volume2 className="h-5 w-5 mr-2 text-purple-400" />
              Generated Audio
            </h2>

            {generatedAudio ? (
              <div className="space-y-4">
                {/* Audio Player */}
                <div className="bg-purple-600/10 rounded-lg p-4 border border-purple-500/20">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <p className="text-white font-medium">{generatedAudio.filename}</p>
                      <p className="text-purple-300 text-sm">
                        {generatedAudio.duration.toFixed(2)}s • {generatedAudio.quality} quality
                      </p>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={handlePlayPause}
                        className="p-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
                      >
                        {isPlaying ? (
                          <Pause className="h-4 w-4 text-white" />
                        ) : (
                          <Play className="h-4 w-4 text-white" />
                        )}
                      </button>
                      
                      <button
                        onClick={handleDownload}
                        className="p-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                      >
                        <Download className="h-4 w-4 text-white" />
                      </button>
                    </div>
                  </div>

                  {/* Audio Waveform Placeholder */}
                  <div className="audio-waveform mb-4" />
                  
                  <audio
                    ref={audioRef}
                    src={generatedAudio.url}
                    onPlay={() => setIsPlaying(true)}
                    onPause={() => setIsPlaying(false)}
                    onEnded={() => setIsPlaying(false)}
                  />

                  {/* Audio Quality Info */}
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-black/20 rounded-lg p-3">
                      <p className="text-gray-400">Format</p>
                      <p className="text-white font-medium">{generatedAudio.format.toUpperCase()}</p>
                    </div>
                    <div className="bg-black/20 rounded-lg p-3">
                      <p className="text-gray-400">Bit Depth</p>
                      <p className="text-white font-medium">{generatedAudio.bitDepth}-bit</p>
                    </div>
                    <div className="bg-black/20 rounded-lg p-3">
                      <p className="text-gray-400">Mastering</p>
                      <p className="text-white font-medium">{generatedAudio.mastering ? 'Applied' : 'None'}</p>
                    </div>
                    <div className="bg-black/20 rounded-lg p-3">
                      <p className="text-gray-400">Quality</p>
                      <p className="text-white font-medium capitalize">{generatedAudio.quality}</p>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="mx-auto w-16 h-16 bg-gray-800/20 rounded-full flex items-center justify-center mb-4">
                  <Music className="h-8 w-8 text-gray-500" />
                </div>
                <p className="text-gray-400">Upload a MIDI file to generate cello audio</p>
              </div>
            )}
          </motion.div>
        </div>

        {/* Training Status */}
        {trainingStatus && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-8 glass-dark rounded-2xl p-6"
          >
            <h3 className="text-lg font-semibold text-white mb-4">Training Status</h3>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-purple-300">Progress</span>
                <span className="text-white font-medium">
                  {Math.round(trainingStatus.progress * 100)}%
                </span>
              </div>
              
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-gradient-to-r from-purple-600 to-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${trainingStatus.progress * 100}%` }}
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-400">Status</p>
                  <p className="text-white font-medium capitalize">{trainingStatus.status}</p>
                </div>
                {trainingStatus.total_samples && (
                  <div>
                    <p className="text-gray-400">Samples Processed</p>
                    <p className="text-white font-medium">{trainingStatus.total_samples}</p>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </main>
    </div>
  );
};

export default DDSPNeuralCello;