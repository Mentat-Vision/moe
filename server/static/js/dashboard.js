class Dashboard {
    constructor() {
        this.cameras = new Map();
        this.updateInterval = null;
        this.statusCheckInterval = null;
        this.yoloStatsInterval = null;
        this.blipStatsInterval = null;
        this.socket = null;
        this.detectionHistory = new Map(); // Store detection history per camera
        this.captionHistory = new Map(); // Store caption history per camera
        this.availableModels = []; // Store available BLIP models
        this.currentModel = null; // Store current BLIP model info
        this.init();
    }

    init() {
        this.updateConnectionStatus('connecting');
        this.startStatusCheck();
        this.loadCameras();
        this.initWebSocket();
        
        // Set up periodic updates - reduced interval for smoother updates
        this.updateInterval = setInterval(() => {
            this.updateCameraFeeds();
        }, 100);
        
        // Set up YOLO stats updates
        this.yoloStatsInterval = setInterval(() => {
            this.updateYoloStats();
        }, 2000);
        
        // Set up toggle buttons
        this.setupToggleBoxesButton();
        this.setupToggleYoloButton();
        this.setupToggleBlipButton();
        this.updateYoloStatus();
        
        // Set up BLIP model selector
        this.loadBlipModels();
        this.updateBlipStatus();
        this.setupModelSwitcher();
    }

    initWebSocket() {
        // Connect to the same server using SocketIO
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('WebSocket connected for video streaming');
            // Request streams for all existing cameras
            this.cameras.forEach((element, cameraId) => {
                this.socket.emit('request_video_stream', { camera_id: cameraId });
            });
        });

        this.socket.on('disconnect', () => {
            console.log('WebSocket disconnected');
        });

        this.socket.on('video_frame', (data) => {
            const { camera_id, frame_data } = data;
            this.updateVideoFrame(camera_id, frame_data);
        });

        this.socket.on('detection_result', (data) => {
            this.handleDetectionResult(data);
        });

        this.socket.on('caption_result', (data) => {
            this.handleCaptionResult(data);
        });
    }

    updateVideoFrame(cameraId, frameData) {
        const cameraElement = this.cameras.get(cameraId);
        if (cameraElement && document.visibilityState === 'visible') {  // Only update if tab is visible
            const img = cameraElement.querySelector('.video-container img');
            if (img) {
                // Use requestAnimationFrame for smoother rendering
                requestAnimationFrame(() => {
                    // Handle binary data directly
                    const blob = new Blob([frameData], { type: 'image/jpeg' });
                    
                    // Revoke previous object URL to prevent memory leaks
                    if (img.currentBlobUrl) {
                        URL.revokeObjectURL(img.currentBlobUrl);
                    }
                    
                    img.currentBlobUrl = URL.createObjectURL(blob);
                    img.src = img.currentBlobUrl;
                    img.style.display = 'block';
                    img.nextElementSibling.style.display = 'none';
                });
            }
        }
    }

    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connection-status');
        statusElement.textContent = status === 'connected' ? 'Connected' : 
                                   status === 'connecting' ? 'Connecting...' : 'Disconnected';
        statusElement.className = status;
    }

    async loadCameras() {
        try {
            const response = await fetch('/api/cameras');
            if (response.ok) {
                const cameras = await response.json();
                this.updateCameraGrid(cameras);
                this.updateConnectionStatus('connected');
                this.updateCameraCount(cameras.length);
            } else {
                throw new Error('Failed to load cameras');
            }
        } catch (error) {
            console.error('Error loading cameras:', error);
            this.updateConnectionStatus('disconnected');
        }
    }

    updateCameraGrid(cameras) {
        const grid = document.getElementById('camera-grid');
        const currentCameras = new Set();

        cameras.forEach(camera => {
            currentCameras.add(camera.id);
            
            if (!this.cameras.has(camera.id)) {
                // Create new camera feed
                const cameraElement = this.createCameraElement(camera);
                grid.appendChild(cameraElement);
                this.cameras.set(camera.id, cameraElement);
            } else {
                // Update existing camera status
                this.updateCameraStatus(camera);
            }
        });

        // Remove cameras that are no longer active
        this.cameras.forEach((element, cameraId) => {
            if (!currentCameras.has(cameraId)) {
                element.remove();
                this.cameras.delete(cameraId);
            }
        });
    }

    createCameraElement(camera) {
        const cameraDiv = document.createElement('div');
        cameraDiv.className = 'camera-feed';
        cameraDiv.dataset.cameraId = camera.id;

        cameraDiv.innerHTML = `
            <div class="camera-header">
                <span class="camera-name">${camera.name}</span>
                <span class="camera-status ${camera.status}">${camera.status}</span>
            </div>
            <div class="video-container">
                <img alt="${camera.name}" style="display: none;" 
                     onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                <div class="no-signal">
                    <div class="loading"></div>
                    <p>No Signal</p>
                </div>
            </div>
            <div class="camera-info">
                <p>Status: <span class="status-text">${camera.status}</span></p>
                <p>FPS: <span class="fps-text">${camera.fps || 0}</span></p>
                <p>Last Update: <span class="last-update">${this.formatTimestamp(camera.last_update)}</span></p>
            </div>
            <div class="detection-info">
                <h4>Latest Detections</h4>
                <div class="detection-list" id="detection-${camera.id}">
                    <p>No detections yet</p>
                </div>
            </div>
            <div class="caption-info">
                <h4>Live Caption</h4>
                <div class="caption-text" id="caption-${camera.id}">
                    <p>No caption yet</p>
                </div>
            </div>
        `;

        // Add click handler for fullscreen
        cameraDiv.addEventListener('click', () => {
            this.toggleFullscreen(cameraDiv);
        });

        // Request video stream for this camera
        if (this.socket && this.socket.connected) {
            this.socket.emit('request_video_stream', { camera_id: camera.id });
        }

        return cameraDiv;
    }

    updateCameraStatus(camera) {
        const cameraElement = this.cameras.get(camera.id);
        if (cameraElement) {
            const statusElement = cameraElement.querySelector('.camera-status');
            const statusText = cameraElement.querySelector('.status-text');
            const fpsText = cameraElement.querySelector('.fps-text');
            const lastUpdateElement = cameraElement.querySelector('.last-update');

            statusElement.className = `camera-status ${camera.status}`;
            statusElement.textContent = camera.status;
            statusText.textContent = camera.status;
            fpsText.textContent = camera.fps || 0;
            lastUpdateElement.textContent = this.formatTimestamp(camera.last_update);
        }
    }

    updateCameraFeeds() {
        // The video feeds are automatically updated via MJPEG streams
        // This method can be used for additional updates if needed
    }

    updateCameraCount(count) {
        document.getElementById('camera-count').textContent = `Cameras: ${count}`;
    }

    formatTimestamp(timestamp) {
        if (!timestamp) return 'Unknown';
        const date = new Date(timestamp * 1000);
        return date.toLocaleTimeString();
    }

    toggleFullscreen(cameraElement) {
        if (cameraElement.classList.contains('fullscreen')) {
            cameraElement.classList.remove('fullscreen');
            document.body.style.overflow = '';
        } else {
            // Exit any existing fullscreen
            document.querySelectorAll('.camera-feed.fullscreen').forEach(el => {
                el.classList.remove('fullscreen');
            });
            
            cameraElement.classList.add('fullscreen');
            document.body.style.overflow = 'hidden';
        }
    }

    handleDetectionResult(data) {
        const { camera_name, objects, timestamp, fps, detection_count, gpu_id } = data;
        
        // Store detection history
        if (!this.detectionHistory.has(camera_name)) {
            this.detectionHistory.set(camera_name, []);
        }
        
        const history = this.detectionHistory.get(camera_name);
        history.push(data);
        
        // Keep only last 10 detections
        if (history.length > 10) {
            history.shift();
        }
        
        // Update detection display
        this.updateDetectionDisplay(camera_name, data);
    }

    handleCaptionResult(data) {
        const { camera_name, caption, timestamp, fps, gpu_id } = data;
        
        // Store caption history
        if (!this.captionHistory.has(camera_name)) {
            this.captionHistory.set(camera_name, []);
        }
        
        const history = this.captionHistory.get(camera_name);
        history.push(data);
        
        // Keep only last 5 captions
        if (history.length > 5) {
            history.shift();
        }
        
        // Update caption display
        this.updateCaptionDisplay(camera_name, data);
    }

    updateDetectionDisplay(cameraId, detectionData) {
        const detectionElement = document.getElementById(`detection-${cameraId}`);
        if (detectionElement) {
            const { objects, timestamp, detection_count, gpu_id } = detectionData;
            
            if (objects.length === 0) {
                detectionElement.innerHTML = '<p>No objects detected</p>';
            } else {
                const objectSummary = this.groupObjectsByClass(objects);
                const detectionTime = new Date(timestamp).toLocaleTimeString();
                
                detectionElement.innerHTML = `
                    <div class="detection-summary">
                        <p><strong>Time:</strong> ${detectionTime}</p>
                        <p><strong>GPU:</strong> ${gpu_id} | <strong>Objects:</strong> ${detection_count}</p>
                        <div class="objects-list">
                            ${Object.entries(objectSummary).map(([className, count]) => 
                                `<span class="object-tag">${className} (${count})</span>`
                            ).join('')}
                        </div>
                    </div>
                `;
            }
        }
    }

    updateCaptionDisplay(cameraId, captionData) {
        const captionElement = document.getElementById(`caption-${cameraId}`);
        if (captionElement) {
            const { caption, timestamp, gpu_id } = captionData;
            const captionTime = new Date(timestamp).toLocaleTimeString();
            
            captionElement.innerHTML = `
                <div class="caption-content">
                    <p class="caption-text-main">"${caption}"</p>
                    <div class="caption-meta">
                        <span>Time: ${captionTime}</span>
                        <span>GPU: ${gpu_id}</span>
                    </div>
                </div>
            `;
        }
    }

    groupObjectsByClass(objects) {
        const summary = {};
        objects.forEach(obj => {
            const className = obj.class_name || 'unknown';
            summary[className] = (summary[className] || 0) + 1;
        });
        return summary;
    }

    async updateYoloStats() {
        try {
            const response = await fetch('/api/yolo/stats');
            if (response.ok) {
                const stats = await response.json();
                this.displayYoloStats(stats);
            }
        } catch (error) {
            console.error('Error fetching YOLO stats:', error);
        }
    }


    displayYoloStats(stats) {
        const statsContent = document.getElementById('yolo-stats-content');
        if (Object.keys(stats).length === 0) {
            statsContent.innerHTML = '<p>No camera processing stats available</p>';
            return;
        }

        const statsHtml = Object.entries(stats).map(([cameraId, stat]) => `
            <div class="camera-stat">
                <h4>${cameraId}</h4>
                <div class="stat-grid">
                    <span>FPS: ${stat.fps}</span>
                    <span>GPU: ${stat.assigned_gpu}</span>
                    <span>Queue: ${stat.queue_size}</span>
                    <span>Objects: ${stat.recent_object_count}</span>
                </div>
            </div>
        `).join('');

        statsContent.innerHTML = statsHtml;
    }


    setupToggleBoxesButton() {
        const toggleBtn = document.getElementById('toggle-boxes-btn');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', async () => {
                try {
                    const response = await fetch('/api/yolo/toggle_boxes', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        toggleBtn.textContent = result.draw_boxes ? 'Hide Boxes' : 'Show Boxes';
                        toggleBtn.style.background = result.draw_boxes ? '#cc2936' : '#333333';
                        console.log(result.message);
                    }
                } catch (error) {
                    console.error('Error toggling bounding boxes:', error);
                }
            });
        }
    }

    setupToggleYoloButton() {
        const toggleBtn = document.getElementById('toggle-yolo-btn');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', async () => {
                try {
                    const response = await fetch('/api/yolo/toggle', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        this.updateYoloButtonState(toggleBtn, result.enabled);
                        console.log(result.message);
                    }
                } catch (error) {
                    console.error('Error toggling YOLO:', error);
                }
            });
        }
    }

    async updateYoloStatus() {
        try {
            const response = await fetch('/api/yolo/status');
            if (response.ok) {
                const status = await response.json();
                const toggleBtn = document.getElementById('toggle-yolo-btn');
                if (toggleBtn) {
                    this.updateYoloButtonState(toggleBtn, status.enabled);
                }
            }
        } catch (error) {
            console.error('Error fetching YOLO status:', error);
        }
    }

    setupToggleBlipButton() {
        const toggleBtn = document.getElementById('toggle-blip-btn');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', async () => {
                try {
                    const response = await fetch('/api/blip/toggle', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        this.updateBlipButtonState(toggleBtn, result.enabled);
                        console.log(result.message);
                    }
                } catch (error) {
                    console.error('Error toggling BLIP:', error);
                }
            });
        }
    }

    updateYoloButtonState(button, enabled) {
        if (enabled) {
            button.textContent = 'YOLO ON';
            button.className = 'toggle-btn yolo-enabled';
        } else {
            button.textContent = 'YOLO OFF';
            button.className = 'toggle-btn yolo-disabled';
        }
    }

    updateBlipButtonState(button, enabled) {
        if (enabled) {
            button.textContent = 'BLIP ON';
            button.className = 'toggle-btn blip-enabled';
        } else {
            button.textContent = 'BLIP OFF';
            button.className = 'toggle-btn blip-disabled';
        }
    }

    async loadBlipModels() {
        try {
            const response = await fetch('/api/blip/models');
            if (response.ok) {
                const data = await response.json();
                this.availableModels = data.models;
                this.populateModelSelector();
            }
        } catch (error) {
            console.error('Error loading BLIP models:', error);
        }
    }

    populateModelSelector() {
        const selector = document.getElementById('blip-model-select');
        if (selector && this.availableModels.length > 0) {
            selector.innerHTML = '';
            
            this.availableModels.forEach(model => {
                const option = document.createElement('option');
                option.value = model.model_name;
                option.textContent = `${model.name} (${model.type.toUpperCase()})`;
                option.title = model.description;
                selector.appendChild(option);
            });
            
            // Set current model as selected if we have it
            if (this.currentModel) {
                selector.value = this.currentModel.model_name;
            }
        }
    }

    async updateBlipStatus() {
        try {
            const response = await fetch('/api/blip/status');
            if (response.ok) {
                const status = await response.json();
                this.currentModel = status;
                
                // Update toggle button
                const toggleBtn = document.getElementById('toggle-blip-btn');
                if (toggleBtn) {
                    this.updateBlipButtonState(toggleBtn, status.enabled);
                }
                
                // Update model selector
                const selector = document.getElementById('blip-model-select');
                if (selector && status.model_name) {
                    selector.value = status.model_name;
                }
            }
        } catch (error) {
            console.error('Error fetching BLIP status:', error);
        }
    }

    setupModelSwitcher() {
        const switchBtn = document.getElementById('switch-model-btn');
        const selector = document.getElementById('blip-model-select');
        
        if (switchBtn && selector) {
            switchBtn.addEventListener('click', async () => {
                const selectedModel = selector.value;
                if (!selectedModel) return;
                
                // Disable button during switch
                switchBtn.disabled = true;
                switchBtn.textContent = 'Switching...';
                
                try {
                    const response = await fetch('/api/blip/switch_model', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            model_name: selectedModel
                        })
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        if (result.success) {
                            console.log(`Successfully switched to ${result.model_type.toUpperCase()} model: ${result.new_model}`);
                            
                            // Update current model info
                            await this.updateBlipStatus();
                            
                            // Show success feedback
                            switchBtn.textContent = 'Switched!';
                            setTimeout(() => {
                                switchBtn.textContent = 'Switch';
                            }, 2000);
                        } else {
                            throw new Error(result.error || 'Failed to switch model');
                        }
                    } else {
                        throw new Error('Server error during model switch');
                    }
                } catch (error) {
                    console.error('Error switching BLIP model:', error);
                    alert(`Error switching model: ${error.message}`);
                    switchBtn.textContent = 'Switch';
                } finally {
                    switchBtn.disabled = false;
                }
            });
        }
    }

    startStatusCheck() {
        this.statusCheckInterval = setInterval(() => {
            this.loadCameras();
        }, 2000); // Check every 2 seconds for more responsive updates
    }

    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
        }
        if (this.yoloStatsInterval) {
            clearInterval(this.yoloStatsInterval);
        }
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.dashboard) {
        window.dashboard.destroy();
    }
});

// Handle visibility change to pause/resume updates
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, could pause updates here
    } else {
        // Page is visible again, refresh cameras
        if (window.dashboard) {
            window.dashboard.loadCameras();
        }
    }
});