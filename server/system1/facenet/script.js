let updateInterval;
let videoInterval;

function startAutoRefresh() {
    // Update stable faces every 3 seconds (less frequent for stability)
    updateInterval = setInterval(refreshCurrentFaces, 3000);
    // Update registered faces every 10 seconds
    setInterval(refreshRegisteredFaces, 10000);
    // Update models every 30 seconds
    setInterval(refreshModels, 30000);
    // Update video feed every 100ms for smooth playback
    videoInterval = setInterval(updateVideoFeed, 100);
}

async function updateVideoFeed() {
    try {
        const response = await fetch('/api/video_frame');
        const data = await response.json();
        
        if (data.frame) {
            const canvas = document.getElementById('video-canvas');
            const ctx = canvas.getContext('2d');
            
            const img = new Image();
            img.onload = function() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = 'data:image/jpeg;base64,' + data.frame;
        }
    } catch (error) {
        console.error('Error updating video feed:', error);
    }
}

async function refreshCurrentFaces() {
    try {
        document.getElementById('refresh-indicator').style.opacity = '1';
        
        const response = await fetch('/api/current_faces');
        const data = await response.json();
        
        displayCurrentFaces(data.faces);
        
        document.getElementById('detected-count').textContent = data.faces.length;
        document.getElementById('registered-count').textContent = data.total_registered;
        document.getElementById('tracks-count').textContent = data.active_tracks;
        
        setTimeout(() => {
            document.getElementById('refresh-indicator').style.opacity = '0.7';
        }, 200);
        
    } catch (error) {
        console.error('Error refreshing current faces:', error);
    }
}

async function refreshRegisteredFaces() {
    try {
        const response = await fetch('/api/faces');
        const faces = await response.json();
        displayRegisteredFaces(faces);
    } catch (error) {
        console.error('Error refreshing registered faces:', error);
    }
}

function displayCurrentFaces(faces) {
    const container = document.getElementById('current-faces');
    
    if (!faces || faces.length === 0) {
        container.innerHTML = '<div class="no-data">No stable faces detected yet</div>';
        return;
    }
    
    container.innerHTML = '';
    
    faces.forEach(face => {
        const isRegistered = face.is_registered;
        const stability = face.stability || 1;
        const stabilityPercent = Math.min(100, (stability / 50) * 100);
        const autoRegisterPercent = Math.min(100, (stability / 25) * 100); // Show progress to auto-registration
        
        const card = document.createElement('div');
        card.className = `face-card ${isRegistered ? 'registered' : 'unregistered'}`;
        
        card.innerHTML = `
            <div class="face-id">${face.id}</div>
            <div class="face-name">${face.name}</div>
            <div class="status-badge ${isRegistered ? 'status-registered' : 'status-unregistered'}">
                ${isRegistered ? 'Auto-Registered' : 'Tracking...'}
            </div>
            <div class="face-info">
                Stability: ${stability}/50 ${!isRegistered ? `(Auto-register at 25)` : ''}
            </div>
            <div class="stability-bar">
                <div class="stability-fill" style="width: ${stabilityPercent}%"></div>
            </div>
            ${!isRegistered ? `
                <div class="auto-register-bar" style="margin-top: 5px;">
                    <div style="font-size: 0.8em; color: #aaa; margin-bottom: 3px;">Auto-registration progress:</div>
                    <div style="width: 100%; height: 3px; background: #222; border-radius: 2px; overflow: hidden;">
                        <div style="height: 100%; background: linear-gradient(90deg, #333, #ff6600, #ff0000); width: ${autoRegisterPercent}%; transition: width 0.3s ease;"></div>
                    </div>
                </div>
            ` : ''}
        `;
        
        container.appendChild(card);
    });
}

function displayRegisteredFaces(faces) {
    const container = document.getElementById('registered-faces');
    
    if (!faces || Object.keys(faces).length === 0) {
        container.innerHTML = '<div class="no-data">No faces registered yet</div>';
        return;
    }
    
    container.innerHTML = '';
    
    Object.entries(faces).forEach(([faceId, faceData]) => {
        const card = document.createElement('div');
        card.className = 'face-card registered';
        
        card.innerHTML = `
            <img src="faces/${faceId}.jpg" class="face-image" alt="${faceData.name}" 
                 onerror="this.style.display='none';">
            <div class="face-id">${faceId}</div>
            <div class="face-name">${faceData.name}</div>
            <div class="status-badge status-registered">Registered</div>
            <div class="timestamp">
                ${new Date(faceData.registered_at).toLocaleDateString()}
            </div>
            
            <input type="text" id="edit-${faceId}" value="${faceData.name}" />
            <br>
            <button class="btn" onclick="updateName('${faceId}')">Update</button>
            <button class="btn btn-primary" onclick="deleteFace('${faceId}')" style="margin-left: 5px;">Delete</button>
        `;
        
        container.appendChild(card);
    });
}

// Note: Manual registration removed - faces are auto-registered with temp IDs
// Use the registered database section to rename faces

async function updateName(faceId) {
    const input = document.getElementById(`edit-${faceId}`);
    const newName = input ? input.value.trim() : '';
    
    if (!newName) {
        console.log('No name entered');
        return;
    }
    
    await updateFaceName(faceId, newName);
}

async function updateFaceName(faceId, newName) {
    try {
        const response = await fetch('/api/rename_face', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({face_id: faceId, new_name: newName})
        });
        
        const result = await response.json();
        
        if (result.success) {
            console.log(`Successfully updated: ${newName}`);
            refreshAll();
        } else {
            console.error('Failed to update name');
        }
    } catch (error) {
        console.error('Error:', error.message);
    }
}

function refreshAll() {
    refreshCurrentFaces();
    refreshRegisteredFaces();
    refreshModels();
}

// Initialize
window.addEventListener('load', () => {
    refreshAll();
    startAutoRefresh();
});

// Handle visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        if (updateInterval) clearInterval(updateInterval);
        if (videoInterval) clearInterval(videoInterval);
    } else {
        startAutoRefresh();
    }
});

// Model management functions
async function refreshModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        const select = document.getElementById('model-select');
        const currentInfo = document.getElementById('current-model-info');
        
        // Clear current options
        select.innerHTML = '<option value="basic">Basic Tracking</option>';
        
        // Add available models
        data.available_models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model.charAt(0).toUpperCase() + model.slice(1);
            if (model === data.current_model) {
                option.selected = true;
            }
            select.appendChild(option);
        });
        
        // Update current model info
        if (data.use_embeddings) {
            currentInfo.textContent = `Current: ${data.current_model}`;
        } else {
            currentInfo.textContent = 'Current: Basic Tracking';
        }
        
    } catch (error) {
        console.error('Error refreshing models:', error);
    }
}

async function switchModel() {
    const select = document.getElementById('model-select');
    const modelName = select.value;
    
    if (modelName === 'basic') {
        console.log('Basic tracking mode - no model switching needed');
        return;
    }
    
    try {
        const response = await fetch('/api/switch_model', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model_name: modelName})
        });
        
        const result = await response.json();
        
        if (result.success) {
            console.log(`Successfully switched to ${modelName}`);
            refreshModels();
        } else {
            console.error(`Failed to switch to ${modelName}. Check if model files are available.`);
        }
    } catch (error) {
        console.error('Error switching model:', error.message);
    }
}


async function deleteFace(faceId) {
    try {
        const response = await fetch('/api/delete_face', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({face_id: faceId})
        });
        
        const result = await response.json();
        
        if (result.success) {
            console.log(`Successfully deleted face: ${faceId}`);
            refreshAll();
        } else {
            console.error(`Failed to delete face: ${faceId}`);
        }
    } catch (error) {
        console.error('Error deleting face:', error.message);
    }
}