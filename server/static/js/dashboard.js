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
        this.faceHistory = new Map(); // Store face recognition history per camera
        this.faceDatabase = new Map(); // Store registered faces
        this.activeFaces = new Set(); // Track currently visible faces
        this.demoNotifications = [];
        this.demoScheduleStarted = false;
        this.cameraHealth = new Map(); // Track camera health badges
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
        this.setupToggleFaceButton();
        this.setupToggleAutoRegisterButton();
        this.setupClearFacesButton();
        this.updateYoloStatus();
        this.updateFaceStatus();
        this.initDemoNotifications();
        this.startSystemStatusAnimation();
        this.initThreatLevelLogic();
        this.setupSimpleViewToggle();
        
        // Set up face grid updates (demo faces from static folder)
        this.faceGridInterval = setInterval(() => {
            this.updateFaceGrid();
        }, 4000);
    }

    setupSimpleViewToggle() {
        const btn = document.getElementById('toggle-simple-view');
        if (!btn) return;
        this.simpleView = false;
        btn.addEventListener('click', () => {
            this.simpleView = !this.simpleView;
            btn.textContent = this.simpleView ? 'Full View' : 'Simple View';
            const middle = document.querySelector('.main-middle');
            const left = document.querySelector('.main-left');
            const chat = document.getElementById('chat-panel');
            const body = document.body;
            if (this.simpleView) {
                if (middle) middle.style.display = 'none';
                if (left) left.style.display = 'none';
                if (chat) chat.style.display = 'block';
                body.classList.add('simple-view');
            } else {
                if (middle) middle.style.display = '';
                if (left) left.style.display = '';
                if (chat) chat.style.display = 'none';
                body.classList.remove('simple-view');
            }
        });
        // Basic chat demo
        const sendBtn = document.getElementById('chat-send');
        const input = document.getElementById('chat-text');
        const messages = document.getElementById('chat-messages');
        if (sendBtn && input && messages) {
            const STORAGE_KEY = 'mentat_chat_history_v1';
            const loadHistory = () => {
                try {
                    const raw = localStorage.getItem(STORAGE_KEY);
                    if (!raw) return [];
                    return JSON.parse(raw);
                } catch { return []; }
            };
            const saveHistory = (items) => {
                try { localStorage.setItem(STORAGE_KEY, JSON.stringify(items)); } catch {}
            };
            const renderHistory = (items) => {
                messages.innerHTML = '';
                items.forEach(it => {
                    const row = document.createElement('div');
                    row.className = `msg ${it.role === 'user' ? 'user' : 'assistant'}`;
                    const bubble = document.createElement('div');
                    bubble.className = 'bubble';
                    bubble.innerHTML = `${it.text}`;
                    row.appendChild(bubble);
                    messages.appendChild(row);
                });
                messages.scrollTop = messages.scrollHeight;
            };

            // Seed realistic past convo if history is empty or trivial
            let history = loadHistory();
            const looksTrivial = history.length < 4 || (history.length === 1 && /ask about today|camera/i.test(history[0].text || ''));
            if (looksTrivial) {
                history = [
                    { role: 'assistant', text: 'Mentat: Evening ops check complete. Ask about vehicles, cameras, or patterns.' },
                    { role: 'user', text: 'Did the white van return near RTSP_101 today?' },
                    { role: 'assistant', text: 'A white van similar to last Tuesday appeared at 19:42 and stayed ~8 minutes; no incidents logged.' },
                    { role: 'user', text: 'Anything unusual at the gate on RTSP_201 this week?' },
                    { role: 'assistant', text: 'No anomalies. Last motion 5 nights ago at 22:17 aligns with scheduled deliveries.' },
                    { role: 'user', text: 'Compare parking occupancy on RTSP_501 vs last Monday.' },
                    { role: 'assistant', text: 'Parking is +1 vehicle vs last Monday, within 30‑day baseline.' },
                    { role: 'user', text: 'How is the corridor on RTSP_301 after 21:00?' },
                    { role: 'assistant', text: 'Consistently low-traffic for the past 14 days. Tonight matches baseline.' },
                    { role: 'user', text: 'Any visibility issues on RTSP_601?' },
                    { role: 'assistant', text: 'Lighting glare slightly increased vs last week; visibility acceptable. Maintenance recommended.' }
                ];
                saveHistory(history);
            }
            renderHistory(history);

            const send = () => {
                const text = (input.value || '').trim();
                if (!text) return;
                input.value = '';
                history.push({ role: 'user', text });
                renderHistory(history);
                saveHistory(history);
                // Demo reply based on simple heuristics
                setTimeout(() => {
                    let reply = 'Historical analysis indicates no anomalies today. Ask about a camera or date.';
                    if (/rtsp_?\d+/i.test(text)) reply = 'No unusual activity for that camera this week. Occupancy and motion are within baseline.';
                    if (/van|vehicle|car/i.test(text)) reply = 'A white van appeared twice in the last 7 days near RTSP_101/RTSP_501 at similar times.';
                    if (/week|today|yesterday/i.test(text)) reply = 'Summary: quiet nights, slightly elevated parking occupancy on two evenings, lighting glare persists on RTSP_601.';
                    history.push({ role: 'assistant', text: `Mentat: ${reply}` });
                    renderHistory(history);
                    saveHistory(history);
                }, 500);
            };
            sendBtn.addEventListener('click', send);
            input.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
        }
    }

    async initDemoNotifications() {
        try {
            const resp = await fetch('/static/data/notifications.json');
            if (!resp.ok) throw new Error('Failed to load notifications JSON');
            const data = await resp.json();
            // Stamp incoming notifications with now
            const now = Date.now();
            this.demoNotifications = data.map((n, idx) => ({
                ...n,
                timestamp: now,
                status: 'new'
            }));
            this.scheduleNotificationsWithin10s();
        } catch (e) {
            console.error('Notification init error', e);
        }
    }

    scheduleNotificationsWithin10s() {
        if (this.demoScheduleStarted || this.demoNotifications.length === 0) return;
        this.demoScheduleStarted = true;
        // Delay first notification by 5s to allow heavy streams to initialize
        const initialDelayMs = 5000;
        // Slow down popping: spread over 20s instead of 10s
        const totalWindowMs = 20000;
        const count = this.demoNotifications.length;
        // Generate random offsets that are increasing within 10s
        const offsets = Array.from({ length: count }, () => Math.random());
        offsets.sort((a, b) => a - b);
        const times = offsets.map(o => initialDelayMs + Math.floor(o * (totalWindowMs - 800))); // add initial delay
        this.demoNotifications.forEach((notif, i) => {
            setTimeout(() => this.pushNotification(notif), times[i]);
        });
    }

    pushNotification(notif) {
        this.renderFeedItem(notif);
        // this.showToast(notif); // Removed toast notifications
        // Update activity and threat level
        this.threatCounter = (this.threatCounter || 0) + 1;
        if (this.activityEl) this.activityEl.textContent = `Threats Today: ${this.threatCounter}`;
        this.updateThreatBadge();
    }

    // Demo animated system status
    startSystemStatusAnimation() {
        const gpuBar = document.getElementById('gpu-bar');
        const memBar = document.getElementById('mem-bar');
        const netBar = document.getElementById('net-bar');
        const gpuText = document.getElementById('gpu-usage');
        const memText = document.getElementById('mem-usage');
        const netText = document.getElementById('net-usage');
        if (!gpuBar || !memBar || !netBar) return;
        setInterval(() => {
            const gpu = 40 + Math.floor(Math.random() * 50);
            const mem = 30 + Math.floor(Math.random() * 60);
            const net = (10 + Math.random() * 90).toFixed(1);
            gpuBar.style.width = gpu + '%';
            memBar.style.width = mem + '%';
            netBar.style.width = Math.min(100, parseFloat(net)) + '%';
            if (gpuText) gpuText.textContent = gpu + '%';
            if (memText) memText.textContent = mem + '%';
            if (netText) netText.textContent = net + ' Mbps';
        }, 1500);
    }

    // Demo threat level indicator reacts to notifications count
    initThreatLevelLogic() {
        this.threatCounter = 0;
        this.activityEl = document.getElementById('activity-counter');
        this.threatEl = document.getElementById('threat-level');
    }

    pushNotification(notif) {
        this.renderFeedItem(notif);
        this.showToast(notif);
        // Update activity and threat level
        this.threatCounter = (this.threatCounter || 0) + 1;
        if (this.activityEl) this.activityEl.textContent = `Threats Today: ${this.threatCounter}`;
        this.updateThreatBadge();
    }

    updateThreatBadge() {
        if (!this.threatEl) return;
        const n = this.threatCounter || 0;
        this.threatEl.classList.remove('low', 'med', 'high');
        if (n < 3) { this.threatEl.classList.add('low'); this.threatEl.textContent = 'Low'; }
        else if (n < 6) { this.threatEl.classList.add('med'); this.threatEl.textContent = 'Medium'; }
        else { this.threatEl.classList.add('high'); this.threatEl.textContent = 'High'; }
    }

    renderFeedItem(notif) {
        const feed = document.getElementById('notif-feed');
        if (!feed) return;
        // Remove empty placeholder
        const empty = feed.querySelector('.notif-empty');
        if (empty) empty.remove();

        const el = document.createElement('div');
        el.className = `notif-item ${notif.severity || 'info'}`;
        el.innerHTML = `
            <div class="notif-title">${notif.title}</div>
            <div class="notif-meta">
                <span>${new Date().toLocaleTimeString()}</span>
                <span>${notif.cameraId || 'Unknown cam'}</span>
                <span>${notif.category || ''}</span>
            </div>
            <div class="notif-message">${notif.message}</div>
            <div class="notif-actions">
                <button class="notif-btn escalate">Escalate</button>
                <button class="notif-btn resolve">Resolve</button>
            </div>
        `;

        const countEl = document.getElementById('notif-count');
        if (countEl) {
            const current = parseInt(countEl.textContent || '0', 10) || 0;
            countEl.textContent = String(current + 1);
        }

        // Wire actions
        const escalateBtn = el.querySelector('.escalate');
        const resolveBtn = el.querySelector('.resolve');
        escalateBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.onEscalate(notif, el);
        });
        resolveBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.onResolve(notif, el);
        });

        // Insert at top
        feed.prepend(el);
    }

    onEscalate(notif, element) {
        // Demo: just mark and show a toast
        notif.status = 'escalated';
        this.showToast({ ...notif, title: `${notif.title} — escalated`, severity: 'critical' });
    }

    onResolve(notif, element) {
        notif.status = 'resolved';
        element.style.opacity = '0.6';
        this.showToast({ ...notif, title: `${notif.title} — resolved`, severity: 'info' });
    }

    showToast(notif) {
        const container = document.getElementById('toast-container');
        if (!container) return;
        const toast = document.createElement('div');
        toast.className = `toast ${notif.severity || 'info'}`;
        toast.innerHTML = `
            <div class="title">${notif.title}</div>
            <div class="message">${notif.message}</div>
            <div class="meta">
                <span>${new Date().toLocaleTimeString()}</span>
                <span>${notif.cameraId || ''}</span>
            </div>
        `;
        container.appendChild(toast);
        // Auto-remove after 4s
        setTimeout(() => {
            toast.style.transition = 'opacity 200ms ease';
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 220);
        }, 4000);
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

        this.socket.on('face_result', (data) => {
            this.handleFaceResult(data);
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

        const health = this.getInitialCameraHealth(camera.id);
        cameraDiv.innerHTML = `
            <div class="camera-header">
                <span class="camera-name">${camera.name}</span>
                <span class="camera-health ${health}">${health === 'optimal' ? 'Optimal' : 'Degraded'}</span>
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
            <div class="face-info">
                <h4>Face Recognition</h4>
                <div class="face-list" id="face-${camera.id}">
                    <p>No faces detected</p>
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

    getInitialCameraHealth(cameraId) {
        if (!this.cameraHealth.has(cameraId)) {
            const state = Math.random() < 0.8 ? 'optimal' : 'degraded';
            this.cameraHealth.set(cameraId, state);
        }
        return this.cameraHealth.get(cameraId);
    }

    updateCameraStatus(camera) {
        const cameraElement = this.cameras.get(camera.id);
        if (cameraElement) {
            const statusElement = cameraElement.querySelector('.camera-status');
            const healthElement = cameraElement.querySelector('.camera-health');
            const statusText = cameraElement.querySelector('.status-text');
            const fpsText = cameraElement.querySelector('.fps-text');
            const lastUpdateElement = cameraElement.querySelector('.last-update');

            statusElement.className = `camera-status ${camera.status}`;
            statusElement.textContent = camera.status;
            statusText.textContent = camera.status;
            fpsText.textContent = camera.fps || 0;
            lastUpdateElement.textContent = this.formatTimestamp(camera.last_update);

            // Occasionally flip health status for demo feel
            if (Math.random() < 0.1) {
                const newHealth = Math.random() < 0.85 ? 'optimal' : 'degraded';
                this.cameraHealth.set(camera.id, newHealth);
                if (healthElement) {
                    healthElement.className = `camera-health ${newHealth}`;
                    healthElement.textContent = newHealth === 'optimal' ? 'Optimal' : 'Degraded';
                }
            }
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
                        toggleBtn.textContent = result.draw_boxes ? 'Hide Overlays' : 'Show Overlays';
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
            button.textContent = 'AI Detection ON';
            button.className = 'toggle-btn yolo-enabled';
        } else {
            button.textContent = 'AI Detection OFF';
            button.className = 'toggle-btn yolo-disabled';
        }
    }

    updateBlipButtonState(button, enabled) {
        if (enabled) {
            button.textContent = 'Scene Analysis ON';
            button.className = 'toggle-btn blip-enabled';
        } else {
            button.textContent = 'Scene Analysis OFF';
            button.className = 'toggle-btn blip-disabled';
        }
    }

    // Face Recognition Methods
    handleFaceResult(data) {
        const { camera_id, faces, timestamp, face_count } = data;
        
        // Store face history
        if (!this.faceHistory.has(camera_id)) {
            this.faceHistory.set(camera_id, []);
        }
        
        const history = this.faceHistory.get(camera_id);
        history.push(data);
        
        // Keep only last 10 face results
        if (history.length > 10) {
            history.shift();
        }
        
        // Update active faces
        if (faces && faces.length > 0) {
            faces.forEach(face => {
                this.activeFaces.add(face.face_id);
            });
        }
        
        // Update face display
        this.updateFaceDisplay(camera_id, data);
    }

    updateFaceDisplay(cameraId, faceData) {
        const faceElement = document.getElementById(`face-${cameraId}`);
        if (faceElement) {
            const { faces, timestamp, face_count } = faceData;
            
            if (faces.length === 0) {
                faceElement.innerHTML = '<p>No faces detected</p>';
            } else {
                const faceTime = new Date(timestamp).toLocaleTimeString();
                
                faceElement.innerHTML = `
                    <div class="face-summary">
                        <p><strong>Time:</strong> ${faceTime}</p>
                        <p><strong>Faces:</strong> ${face_count}</p>
                        <div class="faces-list">
                            ${faces.map(face => {
                                const isRegistered = face.is_registered && !face.name.startsWith('temp_');
                                return `<span class="face-tag ${isRegistered ? 'registered' : 'new'}">${face.name}</span>`;
                            }).join('')}
                        </div>
                    </div>
                `;
            }
        }
    }

    async updateFaceGrid() {
        // Demo: load faces from static faces folder via backend endpoint
        try {
            const response = await fetch('/api/demo/faces');
            if (response.ok) {
                const faces = await response.json();
                this.displayDemoFaceGrid(faces);
                this.updateFaceCount(faces.length);
            }
        } catch (error) {
            console.error('Error fetching demo faces:', error);
        }
    }

    displayDemoFaceGrid(faces) {
        const faceGrid = document.getElementById('face-grid');
        if (!faces || faces.length === 0) {
            faceGrid.innerHTML = '<p>No faces registered yet</p>';
            return;
        }

        const faceHtml = faces.map((f, idx) => {
            const faceId = `demo_${idx}`;
            const isActive = false;
            const cacheBust = Date.now() % 100000;
            const imagePath = (f.url ? encodeURI(f.url) : '/static/mentatRed.png') + `?v=${cacheBust}`;
            const displayName = f.name || `Person ${idx+1}`;
            return `
                <div class="face-item ${isActive ? 'active' : ''}" data-face-id="${faceId}">
                    <div class="face-status ${isActive ? 'active' : ''}"></div>
                    <img src="${imagePath}" alt="${displayName}" onerror="this.src='/static/mentatRed.png'">
                    <div class="face-name">${displayName}</div>
                </div>
            `;
        }).join('');

        faceGrid.innerHTML = faceHtml;
    }

    updateFaceCount(count) {
        document.getElementById('face-count').textContent = `Total Faces: ${count}`;
    }

    setupToggleFaceButton() {
        const toggleBtn = document.getElementById('toggle-face-btn');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', async () => {
                try {
                    const response = await fetch('/api/face/toggle', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        this.updateFaceButtonState(toggleBtn, result.enabled);
                        console.log(result.message);
                    }
                } catch (error) {
                    console.error('Error toggling face recognition:', error);
                }
            });
        }
    }

    setupToggleAutoRegisterButton() {
        const toggleBtn = document.getElementById('toggle-auto-register-btn');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', async () => {
                try {
                    const response = await fetch('/api/face/auto_register/toggle', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        this.updateAutoRegisterButtonState(toggleBtn, result.auto_register);
                        console.log(result.message);
                    }
                } catch (error) {
                    console.error('Error toggling auto-registration:', error);
                }
            });
        }
    }

    setupClearFacesButton() {
        const clearBtn = document.getElementById('clear-faces-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', async () => {
                if (confirm('Are you sure you want to delete all registered faces? This cannot be undone.')) {
                    try {
                        const response = await fetch('/api/face/delete_all', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            }
                        });
                        
                        if (response.ok) {
                            const result = await response.json();
                            if (result.success) {
                                this.updateFaceGrid();
                                console.log('All faces deleted successfully');
                            }
                        }
                    } catch (error) {
                        console.error('Error deleting all faces:', error);
                    }
                }
            });
        }
    }

    async updateFaceStatus() {
        try {
            const response = await fetch('/api/face/status');
            if (response.ok) {
                const status = await response.json();
                const faceBtn = document.getElementById('toggle-face-btn');
                const autoRegBtn = document.getElementById('toggle-auto-register-btn');
                
                if (faceBtn) {
                    this.updateFaceButtonState(faceBtn, status.enabled);
                }
                if (autoRegBtn) {
                    this.updateAutoRegisterButtonState(autoRegBtn, status.auto_register);
                }
            }
        } catch (error) {
            console.error('Error fetching face status:', error);
        }
    }

    updateFaceButtonState(button, enabled) {
        if (enabled) {
            button.textContent = 'Face Recognition ON';
            button.className = 'toggle-btn face-enabled';
        } else {
            button.textContent = 'Face Recognition OFF';
            button.className = 'toggle-btn face-disabled';
        }
    }

    updateAutoRegisterButtonState(button, enabled) {
        if (enabled) {
            button.textContent = 'Auto-Enroll ON';
            button.className = 'toggle-btn auto-register-enabled';
        } else {
            button.textContent = 'Auto-Enroll OFF';
            button.className = 'toggle-btn auto-register-disabled';
        }
    }

    async renameFace(faceId, currentName) {
        const newName = prompt(`Enter new name for "${currentName}":`, currentName);
        if (newName && newName !== currentName) {
            try {
                const response = await fetch('/api/face/update_name', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        face_id: faceId,
                        new_name: newName
                    })
                });
                
                if (response.ok) {
                    const result = await response.json();
                    if (result.success) {
                        this.updateFaceGrid();
                        console.log(`Face renamed to: ${newName}`);
                    }
                }
            } catch (error) {
                console.error('Error renaming face:', error);
            }
        }
    }

    async deleteFace(faceId) {
        if (confirm('Are you sure you want to delete this face?')) {
            try {
                const response = await fetch('/api/face/delete', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        face_id: faceId
                    })
                });
                
                if (response.ok) {
                    const result = await response.json();
                    if (result.success) {
                        this.updateFaceGrid();
                        console.log('Face deleted successfully');
                    }
                }
            } catch (error) {
                console.error('Error deleting face:', error);
            }
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
        if (this.faceGridInterval) {
            clearInterval(this.faceGridInterval);
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