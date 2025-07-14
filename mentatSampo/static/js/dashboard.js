// Global state
let aiModels = {};
let toggleCooldowns = {}; // Track cooldown periods for each toggle

// Toggle AI model globally with improved loading states
async function toggleModel(modelName) {
	// Check if toggle is in cooldown
	if (toggleCooldowns[modelName]) {
		console.log(`⏳ Toggle for ${modelName} is in cooldown`);
		return;
	}

	// Get current state and toggle it
	const currentState = aiModels[modelName].enabled;
	const newState = !currentState;

	// Set cooldown immediately
	toggleCooldowns[modelName] = true;
	const toggleDiv = document.getElementById(`toggle-${modelName}`);
	const switchDiv = document.getElementById(`switch-${modelName}`);

	// Add loading state - grey out and disable
	toggleDiv.classList.remove('enabled', 'disabled');
	toggleDiv.classList.add('loading');
	switchDiv.classList.remove('enabled', 'disabled');
	switchDiv.classList.add('loading');

	console.log(
		`🔄 Toggling ${modelName} to ${newState ? 'enabled' : 'disabled'}...`
	);

	try {
		const response = await fetch(`/api/models/${modelName}/toggle`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ enabled: newState }),
		});

		if (response.ok) {
			const data = await response.json();
			aiModels[modelName].enabled = newState;

			// Remove loading state
			toggleDiv.classList.remove('loading');
			switchDiv.classList.remove('loading');

			// Update to new state with smooth transition
			setTimeout(() => {
				toggleDiv.className = `model-toggle ${
					newState ? 'enabled' : 'disabled'
				}`;
				switchDiv.className = `toggle-switch ${
					newState ? 'enabled' : 'disabled'
				}`;

				console.log(`✅ ${data.message}`);

				// Update all camera displays to reflect the change
				updateAllCameraDisplays();
			}, 100); // Small delay for visual feedback
		} else {
			throw new Error('Failed to toggle model');
		}
	} catch (error) {
		console.error('Error toggling model:', error);

		// Remove loading state
		toggleDiv.classList.remove('loading');
		switchDiv.classList.remove('loading');

		// Revert to previous state
		setTimeout(() => {
			toggleDiv.className = `model-toggle ${
				currentState ? 'enabled' : 'disabled'
			}`;
			switchDiv.className = `toggle-switch ${
				currentState ? 'enabled' : 'disabled'
			}`;
			console.log(
				`❌ Failed to toggle ${modelName}, reverted to previous state`
			);
		}, 100);
	} finally {
		// Set cooldown period (2 seconds)
		setTimeout(() => {
			toggleCooldowns[modelName] = false;
			console.log(`✅ Toggle for ${modelName} is ready again`);
		}, 2000);
	}
}

function updateAllCameraDisplays() {
	// Update all camera displays to reflect model state changes
	cameras.forEach((cameraId) => {
		updateCameraDisplayForModelState(cameraId);
	});
}

function updateCameraDisplayForModelState(cameraId) {
	// Update YOLO display
	const yoloFpsElement = document.getElementById(`yolo-fps-${cameraId}`);
	const personCountElement = document.getElementById(
		`person-count-${cameraId}`
	);
	const detectionsContainer = document.getElementById(`detections-${cameraId}`);

	if (yoloFpsElement) {
		if (!aiModels.yolo.enabled) {
			yoloFpsElement.textContent = '0.0';
			if (personCountElement) personCountElement.textContent = '0';
			if (detectionsContainer) {
				const content =
					detectionsContainer.querySelector('div') || detectionsContainer;
				content.innerHTML = '<div class="no-data">YOLO disabled</div>';
			}
		}
	}

	// Update BLIP display
	const blipFpsElement = document.getElementById(`blip-fps-${cameraId}`);
	const captionElement = document.getElementById(`caption-${cameraId}`);

	if (blipFpsElement) {
		if (!aiModels.blip.enabled) {
			blipFpsElement.textContent = '0.0';
			if (captionElement) captionElement.textContent = 'BLIP disabled';
		}
	}
}

class DashboardClient {
	constructor() {
		this.cameras = new Set();
		this.updateInterval = 2000; // Update every 2 seconds (reduced from 1)
		this.init();
	}

	async init() {
		await this.loadCameras();
		this.startUpdates();
	}

	async loadCameras() {
		try {
			const response = await fetch('/api/cameras');
			const cameras = await response.json();

			if (cameras.length === 0) {
				document.getElementById('cameraGrid').innerHTML =
					'<div class="no-data">No cameras available</div>';
				return;
			}

			this.cameras = new Set(cameras);
			this.renderCameras();
		} catch (error) {
			console.error('Error loading cameras:', error);
			document.getElementById('cameraGrid').innerHTML =
				'<div class="no-data">Error loading cameras</div>';
		}
	}

	renderCameras() {
		const grid = document.getElementById('cameraGrid');
		grid.innerHTML = '';

		this.cameras.forEach((cameraId) => {
			const cameraCard = this.createCameraCard(cameraId);
			grid.appendChild(cameraCard);
		});
	}

	createCameraCard(cameraId) {
		const card = document.createElement('div');
		card.className = 'camera-card';
		card.id = `camera-${cameraId}`;

		card.innerHTML = `
			<div class="camera-header">
				<div class="camera-name">Camera ${cameraId}</div>
				<div class="connection-status disconnected" id="status-${cameraId}">Disconnected</div>
			</div>

			<div class="video-container">
				<img class="video-stream" id="stream-${cameraId}" src="/api/camera/${cameraId}/stream"
					 alt="Camera ${cameraId}" style="display: none;">
				<div class="video-placeholder" id="placeholder-${cameraId}">
					No video feed
				</div>
			</div>

			<div class="camera-stats">
				<div class="stat-group">
					<h3>YOLO Detection</h3>
					<div class="stat-item">
						<span class="stat-label">FPS:</span>
						<span class="stat-value" id="yolo-fps-${cameraId}">-</span>
					</div>
					<div class="stat-item">
						<span class="stat-label">Persons:</span>
						<span class="stat-value" id="person-count-${cameraId}">-</span>
					</div>
				</div>

				<div class="stat-group">
					<h3>BLIP Caption</h3>
					<div class="stat-item">
						<span class="stat-label">FPS:</span>
						<span class="stat-value" id="blip-fps-${cameraId}">-</span>
					</div>
				</div>
			</div>

			<div class="detections-list" id="detections-${cameraId}">
				<h3>Current Detections</h3>
				<div class="no-data" style="padding: 20px;">No detections</div>
			</div>

			<div class="caption-section">
				<h3>Scene Description</h3>
				<div class="caption-text" id="caption-${cameraId}">No caption available</div>
			</div>

			<div class="timestamp" id="timestamp-${cameraId}">Last update: Never</div>
		`;

		// Handle image load events
		const img = card.querySelector(`#stream-${cameraId}`);
		const placeholder = card.querySelector(`#placeholder-${cameraId}`);

		img.onload = () => {
			img.style.display = 'block';
			placeholder.style.display = 'none';
		};

		img.onerror = () => {
			img.style.display = 'none';
			placeholder.style.display = 'flex';
		};

		return card;
	}

	async updateCameraData() {
		for (const cameraId of this.cameras) {
			try {
				const response = await fetch(`/api/camera/${cameraId}/data`);
				const data = await response.json();

				if (data.error) {
					console.log(`❌ Camera ${cameraId} error:`, data.error);
					continue;
				}

				console.log(`🔍 API: Camera ${cameraId} data:`, data);
				this.updateCameraUI(cameraId, data);
			} catch (error) {
				console.error(`Error updating camera ${cameraId}:`, error);
				this.updateConnectionStatus(cameraId, false);
			}
		}
	}

	updateCameraUI(cameraId, data) {
		console.log(`�� Updating camera ${cameraId} with data:`, data);

		// Update connection status
		this.updateConnectionStatus(cameraId, data.connected);

		// Update timestamp
		const timestamp = new Date(data.timestamp * 1000).toLocaleTimeString();
		document.getElementById(
			`timestamp-${cameraId}`
		).textContent = `Last update: ${timestamp}`;

		// Update YOLO data - handle both 'yolo' and 'YOLO' keys
		const yoloData = data.results.yolo || data.results.YOLO || {};
		console.log(`🎯 YOLO data for camera ${cameraId}:`, yoloData);

		// Only update if YOLO is enabled
		if (aiModels.yolo.enabled) {
			document.getElementById(`yolo-fps-${cameraId}`).textContent =
				yoloData.fps || '-';
			document.getElementById(`person-count-${cameraId}`).textContent =
				yoloData.person_count || '0';
			this.updateDetections(cameraId, yoloData.detections || []);
		} else {
			document.getElementById(`yolo-fps-${cameraId}`).textContent = '0.0';
			document.getElementById(`person-count-${cameraId}`).textContent = '0';
			this.updateDetections(cameraId, []);
		}

		// Update BLIP data - handle both 'blip' and 'BLIP' keys
		const blipData = data.results.blip || data.results.BLIP || {};
		console.log(`💬 BLIP data for camera ${cameraId}:`, blipData);

		// Only update if BLIP is enabled
		if (aiModels.blip.enabled) {
			document.getElementById(`blip-fps-${cameraId}`).textContent =
				blipData.fps || '-';
			document.getElementById(`caption-${cameraId}`).textContent =
				blipData.caption || 'No caption available';
		} else {
			document.getElementById(`blip-fps-${cameraId}`).textContent = '0.0';
			document.getElementById(`caption-${cameraId}`).textContent =
				'BLIP disabled';
		}
	}

	updateConnectionStatus(cameraId, connected) {
		const statusElement = document.getElementById(`status-${cameraId}`);
		if (connected) {
			statusElement.className = 'connection-status connected';
			statusElement.textContent = 'Connected';
		} else {
			statusElement.className = 'connection-status disconnected';
			statusElement.textContent = 'Disconnected';
		}
	}

	updateDetections(cameraId, detections) {
		const container = document.getElementById(`detections-${cameraId}`);
		if (!container) return;

		const content = container.querySelector('div') || container;

		if (detections.length === 0) {
			content.innerHTML = '<div class="no-data">No detections</div>';
			return;
		}

		// Limit to 6 detections to prevent overflow on mobile
		const limitedDetections = detections.slice(0, 6);

		content.innerHTML = limitedDetections
			.map(
				(detection) => `
					<div class="detection-item">
						<span class="detection-class">${detection.class}</span>
						<span class="detection-confidence">${(detection.confidence * 100).toFixed(
							1
						)}%</span>
					</div>
				`
			)
			.join('');

		// Add overflow indicator if there are more detections
		if (detections.length > 6) {
			content.innerHTML += `<div class="detection-item">
				<span class="detection-class" style="color: #8b949e; font-style: italic;">
					+${detections.length - 6} more...
				</span>
			</div>`;
		}
	}

	async updateServerStats() {
		try {
			const response = await fetch('/api/stats');
			const stats = await response.json();

			document.getElementById('connectedClients').textContent =
				stats.connected_clients || '-';
			document.getElementById('serverFps').textContent =
				stats.server_fps || '-';
			document.getElementById('totalFrames').textContent =
				stats.total_frames || '-';
			document.getElementById('uptime').textContent = stats.uptime || '-';
		} catch (error) {
			console.error('Error updating server stats:', error);
		}
	}

	startUpdates() {
		// Update camera data
		setInterval(() => this.updateCameraData(), this.updateInterval);

		// Update server stats less frequently
		setInterval(() => this.updateServerStats(), this.updateInterval * 2);

		// Initial updates
		this.updateCameraData();
		this.updateServerStats();
	}
}

// SocketIO client for real-time stats
class SocketIOStatsClient {
	constructor() {
		this.socket = io();
		this.setupEventHandlers();
		this.subscribedCameras = new Set();
	}

	setupEventHandlers() {
		this.socket.on('connect', () => {
			console.log('📡 Connected to SocketIO server');
		});

		this.socket.on('disconnect', () => {
			console.log('📡 Disconnected from SocketIO server');
		});

		this.socket.on('camera_stats_update', (data) => {
			console.log('📡 Received camera stats update:', data);
			this.updateCameraStatsRealtime(data);
		});

		this.socket.on('cameras_list', (data) => {
			console.log('📹 Available cameras:', data.cameras);
		});
	}

	subscribeToCameraStats(cameraId) {
		if (!this.subscribedCameras.has(cameraId)) {
			this.socket.emit('subscribe_camera', { camera_id: cameraId });
			this.subscribedCameras.add(cameraId);
			console.log(`📡 Subscribed to camera ${cameraId} stats`);
		}
	}

	unsubscribeFromCameraStats(cameraId) {
		if (this.subscribedCameras.has(cameraId)) {
			this.socket.emit('unsubscribe_camera', { camera_id: cameraId });
			this.subscribedCameras.delete(cameraId);
			console.log(`📡 Unsubscribed from camera ${cameraId} stats`);
		}
	}

	updateCameraStatsRealtime(data) {
		const cameraId = data.camera_id;
		console.log(`📡 SocketIO update for camera ${cameraId}:`, data);

		// Update connection status
		const statusElement = document.getElementById(`status-${cameraId}`);
		if (statusElement) {
			if (data.connected) {
				statusElement.className = 'connection-status connected';
				statusElement.textContent = 'Connected';
			} else {
				statusElement.className = 'connection-status disconnected';
				statusElement.textContent = 'Disconnected';
			}
		}

		// Update timestamp
		const timestampElement = document.getElementById(`timestamp-${cameraId}`);
		if (timestampElement) {
			const timestamp = new Date(data.timestamp * 1000).toLocaleTimeString();
			timestampElement.textContent = `Last update: ${timestamp}`;
		}

		// Update YOLO data - only if enabled
		const yoloData = data.results.yolo || data.results.YOLO || {};
		console.log(`🎯 YOLO data for camera ${cameraId}:`, yoloData);

		const yoloFpsElement = document.getElementById(`yolo-fps-${cameraId}`);
		const personCountElement = document.getElementById(
			`person-count-${cameraId}`
		);

		if (yoloFpsElement) {
			if (aiModels.yolo.enabled) {
				yoloFpsElement.textContent = yoloData.fps || '-';
				if (personCountElement)
					personCountElement.textContent = yoloData.person_count || '0';
			} else {
				yoloFpsElement.textContent = '0.0';
				if (personCountElement) personCountElement.textContent = '0';
			}
		}

		// Update BLIP data - only if enabled
		const blipData = data.results.blip || data.results.BLIP || {};
		console.log(`💬 BLIP data for camera ${cameraId}:`, blipData);

		const blipFpsElement = document.getElementById(`blip-fps-${cameraId}`);
		const captionElement = document.getElementById(`caption-${cameraId}`);

		if (blipFpsElement) {
			if (aiModels.blip.enabled) {
				blipFpsElement.textContent = blipData.fps || '-';
				if (captionElement)
					captionElement.textContent =
						blipData.caption || 'No caption available';
			} else {
				blipFpsElement.textContent = '0.0';
				if (captionElement) captionElement.textContent = 'BLIP disabled';
			}
		}

		// Update detections - only if YOLO is enabled
		if (aiModels.yolo.enabled) {
			this.updateDetectionsRealtime(cameraId, yoloData.detections || []);
		} else {
			this.updateDetectionsRealtime(cameraId, []);
		}
	}

	updateDetectionsRealtime(cameraId, detections) {
		const container = document.getElementById(`detections-${cameraId}`);
		if (!container) return;

		const content = container.querySelector('div') || container;

		if (detections.length === 0) {
			content.innerHTML = '<div class="no-data">No detections</div>';
			return;
		}

		// Limit to 6 detections to prevent overflow on mobile
		const limitedDetections = detections.slice(0, 6);

		content.innerHTML = limitedDetections
			.map(
				(detection) => `
					<div class="detection-item">
						<span class="detection-class">${detection.class}</span>
						<span class="detection-confidence">${(detection.confidence * 100).toFixed(
							1
						)}%</span>
					</div>
				`
			)
			.join('');

		// Add overflow indicator if there are more detections
		if (detections.length > 6) {
			content.innerHTML += `<div class="detection-item">
				<span class="detection-class" style="color: #8b949e; font-style: italic;">
					+${detections.length - 6} more...
				</span>
			</div>`;
		}
	}
}

// Global SocketIO client
let socketIOClient;

// Enhanced DashboardClient with SocketIO integration
const originalDashboardClient = DashboardClient;
DashboardClient = class extends originalDashboardClient {
	constructor() {
		super();
		this.socketIOClient = new SocketIOStatsClient();
	}

	renderCameras() {
		super.renderCameras();

		// Subscribe to all camera stats via SocketIO
		this.cameras.forEach((cameraId) => {
			this.socketIOClient.subscribeToCameraStats(cameraId);
		});
	}
};

// Live resolution control
async function updateResolution(setting, value) {
	try {
		console.log(
			`🔄 Updating resolution: ${setting} = ${value} (type: ${typeof value})`
		);

		// Update the input value if it's the processing scale
		if (setting === 'PROCESSING_SCALE') {
			const inputElement = document.getElementById('processing-scale');
			if (inputElement) {
				inputElement.value = value;
			}
		}

		// Send update to server
		const requestBody = { setting: setting, value: parseFloat(value) };
		console.log(`📤 Sending request to server:`, requestBody);

		const response = await fetch('/api/resolution/update', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(requestBody),
		});

		console.log(`📥 Server response status: ${response.status}`);

		if (response.ok) {
			const data = await response.json();
			console.log(`✅ Resolution updated: ${data.message}`);

			// If client scale was updated, also update client preview scale
			if (setting === 'CLIENT_PREVIEW_SCALE') {
				// This will be handled by the server to notify clients
				console.log(
					'📡 Client preview scale updated - clients will be notified'
				);
			}
		} else {
			const errorData = await response.json();
			console.error('❌ Failed to update resolution:', errorData);
		}
	} catch (error) {
		console.error('❌ Error updating resolution:', error);
	}
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
	// Initialize AI models from server data
	aiModels = window.aiModelsData || {};
	new DashboardClient();
});
