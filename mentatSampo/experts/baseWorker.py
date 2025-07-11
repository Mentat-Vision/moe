import asyncio
import time
from abc import ABC, abstractmethod

class BaseWorker(ABC):
    """Base class for all expert workers"""
    
    def __init__(self, worker_name, config):
        self.name = worker_name
        self.config = config
        
        # Create async queue for this worker
        self.job_queue = asyncio.Queue(maxsize=100)  # Limit queue size to prevent memory issues
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        print(f"🔧 {self.name} Worker initialized")
    
    async def start(self):
        """Start the worker processing loop"""
        print(f"🚀 Starting {self.name} Worker...")
        await self.initialize_model()
        
        # Start the processing loop
        asyncio.create_task(self.process_loop())
    
    async def process_loop(self):
        """Main processing loop - pulls jobs from queue"""
        while True:
            try:
                # Get job from queue
                job = await self.job_queue.get()
                
                # Process the frame
                result = await self.process_frame(job)
                
                # Update timing
                self.frame_count += 1
                
                # Send result back through the callback
                if "callback" in job:
                    await job["callback"](job["camera_id"], self.name, result)
                
                self.job_queue.task_done()
                
            except Exception as e:
                print(f"❌ {self.name} Worker error: {e}")
                self.job_queue.task_done()
    
    async def add_job(self, camera_id, frame, callback=None):
        """Add a job to the worker's queue"""
        job = {
            "camera_id": camera_id,
            "frame": frame,
            "timestamp": time.time(),
            "callback": callback
        }
        
        try:
            # Non-blocking put - drop frame if queue is full
            self.job_queue.put_nowait(job)
            return True
        except asyncio.QueueFull:
            print(f"⚠️  {self.name} Worker queue full, dropping frame for camera {camera_id}")
            return False
    
    def get_stats(self):
        """Get worker statistics"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            "queue_size": self.job_queue.qsize(),
            "total_frames": self.frame_count,
            "fps": round(fps, 2)
        }
    
    @abstractmethod
    async def initialize_model(self):
        """Initialize the AI model - implement in each worker"""
        pass
    
    @abstractmethod
    async def process_frame(self, job):
        """Process a single frame - implement in each worker"""
        pass 