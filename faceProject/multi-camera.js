// Complete multi-camera JavaScript functionality
let updateInterval;
let cameraInterval;
let editingFaces = new Set();

function startAutoRefresh() {
    updateInterval = setInterval(refreshCombinedFaces, 3000);
    setInterval(refreshModels, 30000);
    cameraInterval = setInterval(updateCameraFeeds, 100);
}

async function updateCameraFeeds() {
    try {
        const response = await fetch('/api/camera_feeds');
        const data = await response.json();
        
        displayCameraFeeds(data.cameras);
        document.getElementById('cameras-count').textContent = data.total_cameras;
        
    } catch (error) {
        console.error('Error updating camera feeds:', error);
    }
}

function displayCameraFeeds(cameras) {
    const container = document.getElementById('camera-grid');
    
    if (!cameras || Object.keys(cameras).length === 0) {
        container.innerHTML = '<div class="no-data">No camera feeds available</div>';
        return;
    }
    
    container.innerHTML = '';
    
    Object.entries(cameras).forEach(([cameraName, cameraData]) => {
        const cameraDiv = document.createElement('div');
        cameraDiv.className = 'camera-feed';
        
        cameraDiv.innerHTML = `
            <div class="camera-feed-header">
                ${cameraName} | FPS: ${cameraData.fps?.toFixed(1) || '0.0'}
            </div>
            <div class="camera-feed-container">
                <img src="data:image/jpeg;base64,${cameraData.frame}" 
                     alt="${cameraName}" />
            </div>
        `;
        
        container.appendChild(cameraDiv);
    });
}

// Core UI functions - Combined faces display
async function refreshCombinedFaces() {
    if (editingFaces.size > 0) return; // Don't refresh while editing
    
    try {
        document.getElementById('refresh-indicator').style.opacity = '1';
        
        // Fetch both current faces and registered faces
        const [currentResponse, registeredResponse] = await Promise.all([
            fetch('/api/current_faces'),
            fetch('/api/faces')
        ]);
        
        const currentData = await currentResponse.json();
        const registeredData = await registeredResponse.json();
        
        displayCombinedFaces(currentData.faces, registeredData);
        
        document.getElementById('detected-count').textContent = currentData.faces.length;
        document.getElementById('registered-count').textContent = currentData.total_registered;
        document.getElementById('tracks-count').textContent = currentData.active_tracks;
        
        setTimeout(() => {
            document.getElementById('refresh-indicator').style.opacity = '0.7';
        }, 500);
        
    } catch (error) {
        console.error('Error refreshing combined faces:', error);
    }
}

function displayCombinedFaces(currentFaces, registeredFaces) {
    const container = document.getElementById('combined-faces');
    
    if ((!currentFaces || currentFaces.length === 0) && (!registeredFaces || Object.keys(registeredFaces).length === 0)) {
        container.innerHTML = '<div class="no-data">No faces detected or registered</div>';
        return;
    }
    
    container.innerHTML = '';
    
    // First, add all currently tracked faces (live)
    if (currentFaces && currentFaces.length > 0) {
        currentFaces.forEach(face => {
            const faceDiv = document.createElement('div');
            const borderColor = face.is_registered ? 'border: 2px solid #00ff00;' : 'border: 2px solid #ff0000;'; // Green for registered, red for unregistered
            faceDiv.className = `face-card ${face.is_registered ? 'registered' : 'unregistered'}`;
            faceDiv.style.cssText += borderColor;
            
            let statusBadge = face.is_registered ? 
                '<span class="status-badge" style="background: #00aa00; color: #fff;">●  LIVE & REGISTERED</span>' :
                '<span class="status-badge status-unregistered">●  LIVE</span>';
            
            let cameraInfo = '';
            if (face.camera_id) {
                cameraInfo = `<div class="face-info">Camera: ${face.camera_id}</div>`;
            }
            if (face.camera_count > 1) {
                cameraInfo += `<div class="face-info">Seen on ${face.camera_count} cameras</div>`;
            }
            
            faceDiv.innerHTML = `
                <div class="face-id">${face.id}</div>
                <div class="face-name">${face.name}</div>
                ${cameraInfo}
                <div class="face-info">Stability: ${face.stability}</div>
                ${statusBadge}
                <div class="stability-bar">
                    <div class="stability-fill" style="width: ${Math.min(100, (face.stability / 25) * 100)}%"></div>
                </div>
            `;
            
            container.appendChild(faceDiv);
        });
    }
    
    // Then add registered faces that are not currently being tracked
    if (registeredFaces && Object.keys(registeredFaces).length > 0) {
        const currentFaceIds = new Set(currentFaces ? currentFaces.map(f => f.id) : []);
        
        Object.entries(registeredFaces).forEach(([faceId, faceData]) => {
            if (!currentFaceIds.has(faceId)) { // Only show if not already shown in live tracking
                const faceDiv = document.createElement('div');
                faceDiv.className = 'face-card registered';
                faceDiv.style.cssText += 'border: 2px solid #666; opacity: 0.7;'; // Gray for offline registered
                
                const imagePath = faceData.image_path ? faceData.image_path : '';
                const imageUrl = imagePath ? `/${imagePath}` : '';
                
                faceDiv.innerHTML = `
                    ${imageUrl ? `<img src="${imageUrl}" class="face-image" alt="${faceData.name}">` : '<div class="face-image" style="background: #333; display: flex; align-items: center; justify-content: center; color: #666;">No Image</div>'}
                    <div class="face-id">${faceId}</div>
                    <input type="text" value="${faceData.name}" 
                           onblur="renameFace('${faceId}', this.value)" 
                           onkeydown="if(event.key==='Enter') this.blur()"
                           onfocus="editingFaces.add('${faceId}')"
                           onblur="editingFaces.delete('${faceId}')">
                    <span class="status-badge" style="background: #666; color: #ccc;">REGISTERED</span>
                    <button class="btn btn-dark" onclick="deleteFace('${faceId}')" style="margin-top: 5px; font-size: 11px; padding: 4px 8px;">Delete</button>
                `;
                
                container.appendChild(faceDiv);
            }
        });
    }
}

async function refreshModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        const select = document.getElementById('model-select');
        select.innerHTML = '';
        
        // Add basic option
        const basicOption = document.createElement('option');
        basicOption.value = 'basic';
        basicOption.textContent = 'OpenCV Haar Cascade (Face Detection Only)';
        if (!data.use_embeddings) {
            basicOption.selected = true;
        }
        select.appendChild(basicOption);
        
        // Add available models
        data.available_models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            if (data.use_embeddings && model === data.current_model) {
                option.selected = true;
            }
            select.appendChild(option);
        });
        
        // Update model info
        const modelInfo = document.getElementById('current-model-info');
        if (modelInfo) {
            if (data.use_embeddings) {
                modelInfo.textContent = `Current: ${data.current_model}`;
            } else {
                modelInfo.textContent = 'Using basic face detection only';
            }
        }
        
    } catch (error) {
        console.error('Error refreshing models:', error);
    }
}

async function switchModel() {
    const select = document.getElementById('model-select');
    const modelName = select.value;
    
    if (modelName === 'basic') {
        console.log('Switched to basic tracking mode');
        return;
    }
    
    try {
        const response = await fetch('/api/switch_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: modelName })
        });
        
        const result = await response.json();
        if (result.success) {
            console.log(`Successfully switched to ${modelName}`);
            setTimeout(refreshModels, 1000);
        } else {
            console.error(`Failed to switch to ${modelName}`);
        }
    } catch (error) {
        console.error('Error switching model:', error);
    }
}

async function renameFace(faceId, newName) {
    if (!newName.trim()) return;
    
    try {
        const response = await fetch('/api/rename_face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                face_id: faceId,
                new_name: newName.trim()
            })
        });
        
        const result = await response.json();
        if (result && result.success) {
            console.log(`Successfully renamed ${faceId} to ${newName}`);
        } else {
            console.error(`Failed to rename ${faceId}`);
        }
    } catch (error) {
        console.error('Error renaming face:', error);
    }
}

async function deleteFace(faceId) {
    if (!confirm(`Delete face ${faceId}?`)) return;
    
    try {
        const response = await fetch('/api/delete_face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ face_id: faceId })
        });
        
        const result = await response.json();
        if (result.success) {
            console.log(`Successfully deleted ${faceId}`);
            setTimeout(refreshCombinedFaces, 500);
        } else {
            console.error(`Failed to delete ${faceId}`);
        }
    } catch (error) {
        console.error('Error deleting face:', error);
    }
}

function refreshAll() {
    refreshCombinedFaces();
    refreshModels();
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Multi-camera interface loaded');
    refreshModels();
    refreshCombinedFaces();
    startAutoRefresh();
});