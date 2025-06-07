import React, { useState, useRef, useEffect } from 'react';
import { Camera, Upload, CheckCircle, AlertTriangle, Aperture, PlusCircle, Save, Map, RefreshCw, Settings, Maximize2, Minimize2, ChevronRight, FileImage, Video, Sliders } from 'lucide-react';
import axios from 'axios';

// Main App Component
export default function CrackDetectorApp() {
  const [activeTab, setActiveTab] = useState('upload');
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [detections, setDetections] = useState([]);
  const [notification, setNotification] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isModelUpdateOpen, setIsModelUpdateOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [sensitivity, setSensitivity] = useState(0.5);
  const [minSize, setMinSize] = useState(30);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const modelInputRef = useRef(null);
  const [taskId, setTaskId] = useState(null); // eslint-disable-line no-unused-vars

  useEffect(() => {
    let videoStream = null;
    if (isCameraActive && videoRef.current) {
      videoStream = videoRef.current.srcObject;
    }
    return () => {
      if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [isCameraActive]);

  const toggleFullscreen = () => setIsFullscreen(!isFullscreen);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
      }
    } catch (err) {
      showNotification('Camera access denied or not available', 'error');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsCameraActive(false);
    }
  };

  const captureAndProcess = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      canvas.toBlob(blob => processImageBlob(blob), 'image/jpeg');
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) setSelectedFile(file);
  };

  const uploadFile = async (file) => {
    try {
      setIsProcessing(true);
      setUploadProgress(0);
      setTaskId(null);

      const baseURL = process.env.NODE_ENV === 'development' ? 'http://localhost:5000' : 'https://underwater-crack-detector.onrender.com';
      const formData = new FormData();
      formData.append('file', file);
      formData.append('threshold', sensitivity.toString());
      formData.append('min_size', minSize.toString());

      const response = await axios.post(`${baseURL}/api/upload`, formData, {
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 90) / progressEvent.total); // Reserve 10% for processing
          setUploadProgress(progress);
        },
      });

      if (response.data.task_id) {
        setTaskId(response.data.task_id);
        pollStatus(response.data.task_id, baseURL);
      }
    } catch (error) {
      console.error('Upload error:', error);
      showNotification(error.response?.data?.error || 'Error uploading file', 'error');
      setIsProcessing(false);
    }
  };

  const processImageBlob = async (blob) => {
    try {
      setIsProcessing(true);
      setUploadProgress(0);
      setTaskId(null);

      const baseURL = process.env.NODE_ENV === 'development' ? 'http://localhost:5000' : 'https://underwater-crack-detector.onrender.com';
      const formData = new FormData();
      formData.append('file', blob, 'camera.jpg');
      formData.append('threshold', sensitivity.toString());
      formData.append('min_size', minSize.toString());

      const response = await axios.post(`${baseURL}/api/upload`, formData);
      if (response.data.task_id) {
        setTaskId(response.data.task_id);
        pollStatus(response.data.task_id, baseURL);
      }
    } catch (error) {
      console.error('Processing error:', error);
      showNotification(error.response?.data?.error || 'Error processing image', 'error');
      setIsProcessing(false);
    }
  };

  const pollStatus = async (task_id, baseURL) => {
    const checkStatus = async () => { // eslint-disable-line no-inner-declarations
      try {
        const response = await axios.get(`${baseURL}/api/status/${task_id}`);
        const data = response.data;
        setUploadProgress(data.progress);

        if (data.progress === 100) {
          if (data.result_url && data.results) {
            setResult(`${baseURL}${data.result_url}`);
            setDetections(data.results);
            showNotification(`Analysis complete! ${data.results.length} cracks detected.`, 'success');
            setActiveTab('results');
          } else if (data.error) {
            showNotification(data.error, 'error');
          }
          setIsProcessing(false);
          setTaskId(null);
        } else if (data.progress < 100) {
          setTimeout(checkStatus, 2000); // Poll every 2 seconds
        }
      } catch (error) {
        console.error('Status check error:', error);
        showNotification('Error checking processing status', 'error');
        setIsProcessing(false);
        setTaskId(null);
      }
    };
    checkStatus();
  };

  const updateModel = async () => {
    const file = modelInputRef.current.files[0];
    if (file && file.name.endsWith('.pth')) {
      try {
        setIsProcessing(true);
        const baseURL = process.env.NODE_ENV === 'development' ? 'http://localhost:5000' : 'https://underwater-crack-detector.onrender.com';
        const formData = new FormData();
        formData.append('model', file);
        const response = await axios.post(`${baseURL}/api/update_model`, formData);
        showNotification(response.data.message || 'Model updated successfully!', 'success');
        setIsModelUpdateOpen(false);
      } catch (error) {
        console.error('Model update error:', error);
        showNotification(error.response?.data?.error || 'Error updating model', 'error');
      } finally {
        setIsProcessing(false);
      }
    } else {
      showNotification('Please select a valid .pth file', 'error');
    }
  };

  const showNotification = (message, type) => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 5000);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) setSelectedFile(e.dataTransfer.files[0]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const getSeverityColor = (severity) => ({
    critical: isDarkMode ? 'bg-red-900 text-red-200' : 'bg-red-100 text-red-800',
    moderate: isDarkMode ? 'bg-yellow-900 text-yellow-200' : 'bg-yellow-100 text-yellow-800',
    minor: isDarkMode ? 'bg-green-900 text-green-200' : 'bg-green-100 text-green-800',
    negligible: isDarkMode ? 'bg-blue-900 text-blue-200' : 'bg-blue-100 text-blue-800',
  }[severity] || (isDarkMode ? 'bg-gray-900 text-gray-200' : 'bg-gray-100 text-gray-800'));

  const getSeverityDotColor = (severity) => ({
    critical: 'bg-red-500',
    moderate: 'bg-yellow-500',
    minor: 'bg-green-500',
    negligible: 'bg-blue-500',
  }[severity] || 'bg-gray-500');

  return (
    <div className={`min-h-screen ${isDarkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-800'}`}>
      <header className={`p-4 ${isDarkMode ? 'bg-gray-800' : 'bg-blue-600'} text-white shadow-md`}>
        <div className="container mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <Aperture className="h-8 w-8" />
            <h1 className="text-xl font-bold">Underwater Crack Detector</h1>
          </div>
          <div className="flex space-x-4 items-center">
            <button onClick={() => setIsDarkMode(!isDarkMode)} className="p-2 rounded-full hover:bg-blue-500 transition-colors">
              {isDarkMode ? '☀️' : '🌙'}
            </button>
            <button onClick={() => setIsSettingsOpen(true)} className="p-2 rounded-full hover:bg-blue-500 transition-colors">
              <Sliders className="h-5 w-5" />
            </button>
            <button onClick={() => setIsModelUpdateOpen(true)} className="p-2 rounded-full hover:bg-blue-500 transition-colors">
              <Settings className="h-5 w-5" />
            </button>
          </div>
        </div>
      </header>

      <main className={`container mx-auto p-4 ${isFullscreen ? 'fixed inset-0 z-50 bg-black' : ''}`}>
        <div className={`flex rounded-t-lg overflow-hidden mb-4 ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow`}>
          <button className={`flex-1 py-3 px-4 flex items-center justify-center gap-2 transition-colors ${activeTab === 'upload' ? isDarkMode ? 'bg-blue-600 text-white' : 'bg-blue-500 text-white' : ''}`} onClick={() => { setActiveTab('upload'); if (isCameraActive) stopCamera(); }}>
            <Upload className="h-5 w-5" />
            <span>Upload</span>
          </button>
          <button className={`flex-1 py-3 px-4 flex items-center justify-center gap-2 transition-colors ${activeTab === 'camera' ? isDarkMode ? 'bg-blue-600 text-white' : 'bg-blue-500 text-white' : ''}`} onClick={() => { setActiveTab('camera'); startCamera(); }}>
            <Camera className="h-5 w-5" />
            <span>Camera</span>
          </button>
          <button className={`flex-1 py-3 px-4 flex items-center justify-center gap-2 transition-colors ${activeTab === 'results' ? isDarkMode ? 'bg-blue-600 text-white' : 'bg-blue-500 text-white' : ''}`} onClick={() => setActiveTab('results')}>
            <Map className="h-5 w-5" />
            <span>Results</span>
          </button>
        </div>

        <div className={`rounded-lg overflow-hidden shadow-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} p-4`}>
          {activeTab === 'upload' && (
            <div className={`border-2 border-dashed rounded-lg p-8 text-center ${isDarkMode ? 'border-gray-600' : 'border-gray-300'}`} onDrop={handleDrop} onDragOver={handleDragOver}>
              <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept="image/*,video/*" className="hidden" />
              {!selectedFile && !isProcessing ? (
                <div className="space-y-4">
                  <div className="mx-auto w-20 h-20 flex items-center justify-center rounded-full bg-blue-100 text-blue-600">
                    <Upload className="h-10 w-10" />
                  </div>
                  <h3 className="text-lg font-semibold">Upload File</h3>
                  <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    Drag and drop an image or video file here, or click to browse
                  </p>
                  <div className="flex flex-col sm:flex-row gap-2 justify-center mt-4">
                    <button onClick={() => fileInputRef.current.click()} className={`px-4 py-2 rounded-lg ${isDarkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-500 hover:bg-blue-600'} text-white flex items-center justify-center gap-2 transition-colors`}>
                      <FileImage className="h-5 w-5" />
                      Select Image
                    </button>
                    <button onClick={() => fileInputRef.current.click()} className={`px-4 py-2 rounded-lg ${isDarkMode ? 'bg-purple-600 hover:bg-purple-700' : 'bg-purple-500 hover:bg-purple-600'} text-white flex items-center justify-center gap-2 transition-colors`}>
                      <Video className="h-5 w-5" />
                      Select Video
                    </button>
                  </div>
                </div>
              ) : isProcessing ? (
                <div className="space-y-4">
                  <div className="w-full max-w-md mx-auto h-2 bg-gray-300 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-500 transition-all duration-300" style={{ width: `${uploadProgress}%` }}></div>
                  </div>
                  <p className="text-sm font-medium">
                    {uploadProgress < 100 ? 'Uploading...' : 'Processing...'} {uploadProgress}%
                  </p>
                  <div className="animate-spin mx-auto">
                    <RefreshCw className="h-8 w-8 text-blue-500" />
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative w-full max-w-lg mx-auto aspect-video bg-gray-200 rounded-lg overflow-hidden">
                    <img src={URL.createObjectURL(selectedFile)} alt="Preview" className="w-full h-full object-contain" />
                  </div>
                  <p className="text-sm font-medium">
                    {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
                  </p>
                  <div className="flex gap-2 justify-center">
                    <button onClick={() => setSelectedFile(null)} className={`px-4 py-2 rounded-lg ${isDarkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-200 hover:bg-gray-300'} transition-colors flex items-center gap-2`}>
                      <RefreshCw className="h-4 w-4" />
                      Change
                    </button>
                    <button onClick={() => uploadFile(selectedFile)} className={`px-4 py-2 rounded-lg ${isDarkMode ? 'bg-green-600 hover:bg-green-700' : 'bg-green-500 hover:bg-green-600'} text-white transition-colors flex items-center gap-2`}>
                      <CheckCircle className="h-4 w-4" />
                      Analyze
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'camera' && (
            <div className="space-y-4">
              <div className={`relative rounded-lg overflow-hidden bg-black ${isFullscreen ? 'fixed inset-0 z-50' : 'aspect-video'}`}>
                <video ref={videoRef} autoPlay playsInline className="w-full h-full object-contain" />
                <canvas ref={canvasRef} className="hidden" />
                <div className="absolute bottom-4 inset-x-0 flex justify-center space-x-4">
                  <button onClick={captureAndProcess} className="p-4 bg-red-500 rounded-full shadow-lg hover:bg-red-600 transition-colors flex items-center justify-center" disabled={!isCameraActive || isProcessing}>
                    <div className="h-6 w-6 rounded-full border-2 border-white"></div>
                  </button>
                </div>
                <button onClick={toggleFullscreen} className="absolute top-4 right-4 p-2 bg-black bg-opacity-50 rounded-full">
                  {isFullscreen ? <Minimize2 className="h-5 w-5 text-white" /> : <Maximize2 className="h-5 w-5 text-white" />}
                </button>
              </div>
              <div className="flex justify-center space-x-4">
                <button onClick={() => { isCameraActive ? stopCamera() : startCamera(); }} className={`px-4 py-2 rounded-lg flex items-center gap-2 ${isCameraActive ? isDarkMode ? 'bg-red-600 hover:bg-red-700' : 'bg-red-500 hover:bg-red-600' : isDarkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-500 hover:bg-blue-600'} text-white transition-colors`}>
                  {isCameraActive ? <> <Camera className="h-5 w-5" /> Stop Camera</> : <> <Camera className="h-5 w-5" /> Start Camera</>}
                </button>
              </div>
            </div>
          )}

          {activeTab === 'results' && (
            <div className="space-y-4">
              {result ? (
                <>
                  <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden shadow-lg">
                    {result.endsWith('.mp4') ? (
                      <video src={result} controls className="w-full h-full object-contain" />
                    ) : (
                      <img src={result} alt="Analysis Result" className="w-full h-full object-contain" />
                    )}
                  </div>
                  <div className={`rounded-lg p-4 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                    <h3 className="text-lg font-semibold mb-2">Analysis Results</h3>
                    {detections.length > 0 ? (
                      <div className="space-y-2">
                        {detections.map(crack => (
                          <div key={crack.id} className={`p-3 rounded-lg flex justify-between items-center ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow`}>
                            <div className="flex items-center gap-2">
                              <div className={`w-3 h-3 rounded-full ${getSeverityDotColor(crack.severity)}`} />
                              <span>Crack #{crack.id} (Size: {crack.position.w.toFixed(2)}x{crack.position.h.toFixed(2)}px, Area: {crack.crack_area_cm2.toFixed(2)} cm²)</span>
                            </div>
                            <div className="flex items-center gap-4">
                              <span className="text-sm"></span>
                              <span className={`text-xs px-2 py-1 rounded-full ${getSeverityColor(crack.severity)}`}>
                                {crack.severity.charAt(0).toUpperCase() + crack.severity.slice(1)}
                              </span>
                              <ChevronRight className="h-4 w-4 text-gray-400" />
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className={`text-center py-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>No significant cracks detected</p>
                    )}
                  </div>
                  <div className="flex justify-center gap-2">
                    <button className={`px-4 py-2 rounded-lg ${isDarkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-500 hover:bg-blue-600'} text-white transition-colors flex items-center gap-2`} onClick={() => {
                      const link = document.createElement('a');
                      link.href = result;
                      link.download = `crack_analysis_${new Date().toISOString().slice(0,10)}.jpg`;
                      document.body.appendChild(link);
                      link.click();
                      document.body.removeChild(link);
                    }}>
                      <Save className="h-4 w-4" /> Save Report
                    </button>
                    <button className={`px-4 py-2 rounded-lg ${isDarkMode ? 'bg-purple-600 hover:bg-purple-700' : 'bg-purple-500 hover:bg-purple-600'} text-white transition-colors flex items-center gap-2`} onClick={() => {
                      setSelectedFile(null);
                      setActiveTab('upload');
                    }}>
                      <PlusCircle className="h-4 w-4" /> New Analysis
                    </button>
                  </div>
                </>
              ) : (
                <div className={`border-2 border-dashed rounded-lg p-8 text-center ${isDarkMode ? 'border-gray-600' : 'border-gray-300'}`}>
                  <div className="mx-auto w-20 h-20 flex items-center justify-center rounded-full bg-gray-100 text-gray-400">
                    <Map className="h-10 w-10" />
                  </div>
                  <h3 className="text-lg font-semibold mt-4">No Results Yet</h3>
                  <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'} mt-2`}>
                    Upload an image or use the camera to detect cracks
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      {isSettingsOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className={`rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} p-6 max-w-md w-full shadow-xl`}>
            <h3 className="text-lg font-semibold mb-4">Detection Settings</h3>
            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-sm font-medium mb-1">Sensitivity (threshold)</label>
                <div className="flex items-center">
                  <input type="range" min="0.1" max="0.9" step="0.1" value={sensitivity} onChange={e => setSensitivity(parseFloat(e.target.value))} className="w-full" />
                  <span className="ml-2 text-sm">{sensitivity.toFixed(1)}</span>
                </div>
                <p className="text-xs text-gray-500 mt-1">Lower values detect more cracks, higher values reduce false positives</p>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Minimum Crack Size</label>
                <div className="flex items-center">
                  <input type="range" min="5" max="100" step="5" value={minSize} onChange={e => setMinSize(parseInt(e.target.value))} className="w-full" />
                  <span className="ml-2 text-sm">{minSize} px</span>
                </div>
                <p className="text-xs text-gray-500 mt-1">Filters out small artifacts that aren't actual cracks</p>
              </div>
            </div>
            <div className="flex justify-end">
              <button onClick={() => setIsSettingsOpen(false)} className={`px-4 py-2 rounded-lg ${isDarkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-500 hover:bg-blue-600'} text-white transition-colors`}>
                Apply Settings
              </button>
            </div>
          </div>
        </div>
      )}

      {isModelUpdateOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className={`rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} p-6 max-w-md w-full shadow-xl`}>
            <h3 className="text-lg font-semibold mb-4">Update Model</h3>
            <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'} mb-4`}>
              Upload a new .pth model file to update the crack detection algorithm
            </p>
            <div className={`border-2 border-dashed rounded-lg p-4 text-center ${isDarkMode ? 'border-gray-600' : 'border-gray-300'} mb-4`}>
              <input type="file" ref={modelInputRef} accept=".pth" className="hidden" />
              <button onClick={() => modelInputRef.current.click()} className={`px-4 py-2 rounded-lg ${isDarkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-500 hover:bg-blue-600'} text-white flex items-center justify-center gap-2 transition-colors mx-auto`}>
                <Upload className="h-4 w-4" /> Select Model File
              </button>
            </div>
            <div className="flex justify-end gap-2 mt-4">
              <button onClick={() => setIsModelUpdateOpen(false)} className={`px-4 py-2 rounded-lg ${isDarkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-200 hover:bg-gray-300'} transition-colors`}>
                Cancel
              </button>
              <button onClick={updateModel} className={`px-4 py-2 rounded-lg ${isDarkMode ? 'bg-green-600 hover:bg-green-700' : 'bg-green-500 hover:bg-green-600'} text-white transition-colors flex items-center gap-2`}>
                <CheckCircle className="h-4 w-4" /> Update
              </button>
            </div>
          </div>
        </div>
      )}

      {notification && (
        <div className={`fixed bottom-4 right-4 p-4 rounded-lg shadow-lg max-w-sm flex items-center gap-3 ${notification.type === 'success' ? isDarkMode ? 'bg-green-800 text-green-100' : 'bg-green-100 text-green-800 border border-green-200' : isDarkMode ? 'bg-red-800 text-red-100' : 'bg-red-100 text-red-800 border border-red-200'}`}>
          {notification.type === 'success' ? <CheckCircle className="h-5 w-5" /> : <AlertTriangle className="h-5 w-5" />}
          <span>{notification.message}</span>
        </div>
      )}
    </div>
  );
}