import React, { useState, useRef } from 'react';
import { Upload, FileVideo, AlertTriangle, CheckCircle, Loader2, Scan, ShieldAlert } from 'lucide-react';

export default function CrimeDetection() {
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisMode, setAnalysisMode] = useState<'gemini' | 'movinet'>('gemini');
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<Array<{
    id: number;
    detected: boolean;
    type?: string;
    confidence?: number;
    timestamp: string;
    frameTime: string;
    details?: string;
  }>>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setVideoUrl(URL.createObjectURL(selectedFile));
      setResults([]);
      setProgress(0);
    }
  };

  const videoRef = useRef<HTMLVideoElement>(null);

  const seekAndCapture = async (time: number): Promise<string | null> => {
    if (!videoRef.current) return null;

    return new Promise((resolve) => {
      const onSeeked = () => {
        videoRef.current?.removeEventListener('seeked', onSeeked);

        const canvas = document.createElement('canvas');
        if (!videoRef.current) { resolve(null); return; }

        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) { resolve(null); return; }

        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL('image/jpeg', 0.95));
      };

      videoRef.current.addEventListener('seeked', onSeeked);
      videoRef.current.currentTime = time;
    });
  };

  const handleAnalyze = async () => {
    if (!file || !videoRef.current) return;

    setAnalyzing(true);
    setProgress(0);
    setResults([]);

    if (analysisMode === 'movinet') {
      try {
        const formData = new FormData();
        formData.append('video', file);

        const response = await fetch('/api/analyze-video', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          if (response.status === 501) {
            // MoViNet not available on Vercel
            setResults([{
              id: Date.now(),
              detected: false,
              type: "Unavailable",
              confidence: 0,
              timestamp: new Date().toLocaleTimeString(),
              frameTime: "00:00",
              details: errorData.suggestion || "MoViNet analysis is not available in cloud deployment. Please switch to Gemini (Image-based) mode."
            }]);
            setProgress(100);
            setAnalyzing(false);
            return;
          }
          throw new Error("Analysis failed");
        }

        const data = await response.json();
        setResults([{
          id: Date.now(),
          detected: data.detected,
          type: data.type,
          confidence: data.confidence,
          timestamp: new Date().toLocaleTimeString(),
          frameTime: "00:00",
          details: data.details
        }]);
        setProgress(100);
      } catch (error) {
        console.error("MoViNet analysis failed", error);
      } finally {
        setAnalyzing(false);
      }
      return;
    }

    // Existing Gemini Logic...
    const duration = videoRef.current.duration;
    const validDuration = (isFinite(duration) && duration > 0) ? duration : 10;

    // Capture keyframes at intervals of 1 second throughout the video
    const timestamps = [];
    // Start from 0.5s to avoid potential black frames at 0.0s
    for (let t = 0.5; t < validDuration; t += 1.0) {
      timestamps.push(t);
    }

    const newResults = [];

    try {
      for (let i = 0; i < timestamps.length; i++) {
        // Check if analysis was cancelled (optional optimization, but good practice)
        if (!videoRef.current) break;

        const baseTime = timestamps[i];

        // Capture a sequence of 5 frames to detect motion (0s, +0.5s, +1.0s, +1.5s, +2.0s)
        const frames = [];
        for (let j = 0; j < 5; j++) {
          const frame = await seekAndCapture(Math.min(baseTime + (j * 0.5), validDuration));
          if (frame) frames.push(frame);
        }

        if (frames.length < 3) continue; // Need at least 3 frames for motion

        // Analyze sequence
        const response = await fetch('/api/analyze-footage', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ images: frames })
        });

        if (!response.ok) {
          console.error(`Analysis failed with status: ${response.status}`);
          continue;
        }

        const text = await response.text();
        let data;
        try {
          data = JSON.parse(text);
        } catch (e) {
          console.error("Failed to parse JSON response:", text.substring(0, 100));
          continue;
        }

        if (data.detected) {
          newResults.push({
            id: i,
            detected: true,
            type: data.type,
            confidence: data.confidence,
            timestamp: new Date().toLocaleTimeString(),
            frameTime: new Date(baseTime * 1000).toISOString().substr(14, 5),
            details: data.details
          });
        }

        setProgress(Math.round(((i + 1) / timestamps.length) * 100));
      }

      setResults(newResults);

    } catch (error) {
      console.error("Analysis failed", error);
    } finally {
      setAnalyzing(false);
      if (videoRef.current) videoRef.current.currentTime = 0;
    }
  };

  return (
    <div className="p-6 space-y-6 h-full overflow-y-auto bg-slate-950 text-slate-200">
      <header className="mb-8 border-b border-slate-800 pb-4 flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <ShieldAlert className="text-red-500" />
            Crime Detection
            <span className={`ml-2 px-2 py-0.5 rounded-full border text-xs font-mono tracking-wider flex items-center gap-1 transition-all ${analysisMode === 'gemini'
                ? 'bg-blue-900/30 border-blue-500/30 text-blue-400'
                : 'bg-purple-900/30 border-purple-500/30 text-purple-400'
              }`}>
              <span className={`w-1.5 h-1.5 rounded-full animate-pulse ${analysisMode === 'gemini' ? 'bg-blue-400' : 'bg-purple-400'
                }`}></span>
              {analysisMode === 'gemini' ? 'POWERED BY GEMINI 3.1 PRO' : 'POWERED BY MOVINET A0'}
            </span>
          </h2>
          <p className="text-slate-500 text-sm">Upload CCTV footage for AI-powered threat analysis</p>
        </div>

        <div className="flex bg-slate-900 p-1 rounded-lg border border-slate-800">
          <button
            onClick={() => setAnalysisMode('gemini')}
            className={`px-3 py-1.5 rounded-md text-xs font-bold transition-all ${analysisMode === 'gemini' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'
              }`}
          >
            GEMINI (Image-based)
          </button>
          <button
            onClick={() => setAnalysisMode('movinet')}
            className={`px-3 py-1.5 rounded-md text-xs font-bold transition-all ${analysisMode === 'movinet' ? 'bg-purple-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'
              }`}
          >
            MOVINET (Video-based)
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload & Preview Section */}
        <div className="space-y-6">
          <div
            className={`border-2 border-dashed rounded-xl overflow-hidden transition-all relative ${file ? 'border-slate-700 bg-black' : 'border-slate-700 hover:border-slate-500 hover:bg-slate-900 p-10 flex flex-col items-center justify-center'
              }`}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                const selectedFile = e.dataTransfer.files[0];
                setFile(selectedFile);
                setVideoUrl(URL.createObjectURL(selectedFile));
                setResults([]);
                setProgress(0);
              }
            }}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="video/*"
              className="hidden"
            />

            {file && videoUrl ? (
              <div className="relative w-full aspect-video group">
                <video
                  ref={videoRef}
                  src={videoUrl}
                  controls
                  className="w-full h-full object-contain"
                  crossOrigin="anonymous"
                />

                {/* Analysis Overlay */}
                {analyzing && (
                  <div className="absolute inset-0 pointer-events-none z-10">
                    <div className="absolute inset-0 bg-blue-900/20 mix-blend-overlay"></div>
                    <div className="absolute top-0 left-0 w-full h-1 bg-blue-500 shadow-[0_0_15px_rgba(59,130,246,1)] animate-scan"></div>
                    <div className="absolute top-4 right-4 font-mono text-xs text-blue-400 bg-black/80 px-2 py-1 rounded">
                      ANALYZING FRAMES... {progress}%
                    </div>
                  </div>
                )}

                <button
                  onClick={() => { setFile(null); setVideoUrl(null); setResults([]); }}
                  className="absolute top-2 right-2 bg-black/50 hover:bg-red-900/80 text-white p-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                  title="Remove Video"
                >
                  <AlertTriangle className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <div className="text-center cursor-pointer" onClick={() => fileInputRef.current?.click()}>
                <Upload className="w-16 h-16 text-slate-600 mx-auto mb-4 group-hover:text-slate-400" />
                <p className="text-lg font-medium text-slate-300">Drag & Drop Video Here</p>
                <p className="text-sm text-slate-500 mt-2">or click to browse</p>
                <p className="text-xs text-slate-600 mt-4">Supports MP4, AVI, MKV</p>
              </div>
            )}
          </div>

          <button
            onClick={handleAnalyze}
            disabled={!file || analyzing}
            className={`w-full py-4 rounded-xl font-bold text-lg tracking-wide transition-all flex items-center justify-center gap-2 ${!file
              ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
              : analyzing
                ? 'bg-blue-600/50 text-blue-200 cursor-wait'
                : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/20'
              }`}
          >
            {analyzing ? (
              <>
                <Loader2 className="animate-spin w-6 h-6" />
                PROCESSING... {progress}%
              </>
            ) : (
              <>
                <Scan className="w-6 h-6" />
                DETECT CRIME
              </>
            )}
          </button>
        </div>

        {/* Results Section */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 relative overflow-hidden flex flex-col h-[600px]">
          {!results.length && !analyzing && (
            <div className="h-full flex flex-col items-center justify-center text-slate-600 opacity-50">
              <Scan className="w-24 h-24 mb-4" />
              <p>Waiting for analysis...</p>
            </div>
          )}

          {analyzing && (
            <div className="h-full flex flex-col items-center justify-center">
              <div className="relative w-32 h-32 mb-8">
                <div className="absolute inset-0 border-4 border-slate-700 rounded-full"></div>
                <div
                  className="absolute inset-0 border-4 border-blue-500 rounded-full border-t-transparent animate-spin"
                  style={{ animationDuration: '1s' }}
                ></div>
                <Scan className="absolute inset-0 m-auto w-12 h-12 text-blue-500 animate-pulse" />
              </div>
              <p className="text-blue-400 font-mono animate-pulse">Scanning frames for anomalies...</p>
              <div className="w-64 h-1 bg-slate-800 mt-6 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-150 ease-out"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
            </div>
          )}

          {results.length > 0 && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500 overflow-y-auto pr-2">
              <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                <h3 className="text-xl font-bold text-white flex items-center gap-2">
                  <AlertTriangle className="text-red-500" />
                  Detections Found ({results.length})
                </h3>
              </div>

              {results.map((result, idx) => (
                <div key={idx} className="bg-slate-950 border border-slate-800 rounded-lg p-4 hover:border-red-900/50 transition-colors">
                  <div className="flex justify-between items-start mb-2">
                    <span className="bg-red-900/30 text-red-400 text-xs font-bold px-2 py-1 rounded uppercase tracking-wider">
                      {result.type}
                    </span>
                    <span className="text-slate-500 font-mono text-xs">
                      {result.frameTime}
                    </span>
                  </div>
                  <p className="text-sm text-slate-300 mb-3">{result.details}</p>
                  <div className="flex items-center justify-between text-xs text-slate-500 font-mono">
                    <span>Confidence: {(result.confidence! * 100).toFixed(1)}%</span>
                    <button
                      onClick={() => {
                        if (videoRef.current) {
                          // Convert frameTime (MM:SS) back to seconds for seeking
                          const [mins, secs] = result.frameTime.split(':').map(Number);
                          videoRef.current.currentTime = mins * 60 + secs;
                          videoRef.current.play();
                        }
                      }}
                      className="text-blue-400 hover:text-blue-300 flex items-center gap-1"
                    >
                      <Scan className="w-3 h-3" /> Review Frame
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {!analyzing && results.length === 0 && file && (
            <div className="h-full flex flex-col items-center justify-center text-green-500/50">
              <CheckCircle className="w-24 h-24 mb-4" />
              <p>No threats detected in analyzed frames.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


