from threading import Lock
import redis
from .router import RegionRouter

class ProjectManager:
    """
    Manages multiple projects, each with their own RegionRouter.
    Automatically switches to next available project when current project's regions are exhausted.
    """
    
    def __init__(self, use_redis=True, redis_host='localhost', redis_port=6379, redis_db=0, key_prefix='project_manager'):
        self.use_redis = use_redis
        self.key_prefix = key_prefix
        self.lock = Lock()
        
        if self.use_redis:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
            self.projects_key = f"{self.key_prefix}:projects"
            self.current_project_key = f"{self.key_prefix}:current_project"
            
            # Initialize current project index if it doesn't exist
            if not self.redis_client.exists(self.current_project_key):
                self.redis_client.set(self.current_project_key, 0)
        else:
            self.redis_client = None
            self.current_project_index = 0
            self.projects_data = {}
        
        # Store project routers
        self.project_routers = {}
        self.project_names = {}  # Maps index to project name
        self.project_indices = {}  # Maps project name to index
        
    def add_project(self, project_index, project_name):
        """Add a new project with the given index and name"""
        with self.lock:
            project_key_prefix = f"{self.key_prefix}:project_{project_index}"
            
            router = RegionRouter(
                use_redis=self.use_redis,
                redis_host=self.redis_client.connection_pool.connection_kwargs['host'] if self.use_redis else 'localhost',
                redis_port=self.redis_client.connection_pool.connection_kwargs['port'] if self.use_redis else 6379,
                redis_db=self.redis_client.connection_pool.connection_kwargs['db'] if self.use_redis else 0,
                key_prefix=project_key_prefix
            )
            
            self.project_routers[project_index] = router
            self.project_names[project_index] = project_name
            self.project_indices[project_name] = project_index
            
            # Store project info in persistent storage
            if self.use_redis:
                projects_data = self.redis_client.hgetall(self.projects_key)
                projects_data[str(project_index)] = project_name
                self.redis_client.hset(self.projects_key, mapping=projects_data)
            else:
                self.projects_data[project_index] = project_name
                
            print(f"Added project {project_index}: '{project_name}'")
    
    def remove_project(self, project_identifier):
        """Remove a project by index or name"""
        with self.lock:
            # Handle both index and name
            if isinstance(project_identifier, str):
                if project_identifier in self.project_indices:
                    project_index = self.project_indices[project_identifier]
                else:
                    print(f"Project '{project_identifier}' not found")
                    return False
            else:
                project_index = project_identifier
                
            if project_index not in self.project_routers:
                print(f"Project index {project_index} not found")
                return False
                
            # Clean up router
            self.project_routers[project_index].close()
            
            # Remove from mappings
            project_name = self.project_names[project_index]
            del self.project_routers[project_index]
            del self.project_names[project_index]
            del self.project_indices[project_name]
            
            # Remove from persistent storage
            if self.use_redis:
                self.redis_client.hdel(self.projects_key, str(project_index))
            else:
                if project_index in self.projects_data:
                    del self.projects_data[project_index]
                    
            print(f"Removed project {project_index}: '{project_name}'")
            return True
    
    def get_current_project_index(self):
        """Get the current project index"""
        if self.use_redis:
            return int(self.redis_client.get(self.current_project_key) or 0)
        else:
            return self.current_project_index
    
    def set_current_project_index(self, index):
        """Set the current project index"""
        if self.use_redis:
            self.redis_client.set(self.current_project_key, index)
        else:
            self.current_project_index = index
    
    def get_next_available_project(self, tokens_needed, exclude_indices=None):
        """
        Find the next project that has enough tokens in any region.
        Returns (project_index, project_name, region, bucket) or (None, None, None, None)
        """
        if exclude_indices is None:
            exclude_indices = set()
            
        project_indices = [idx for idx in self.project_routers.keys() if idx not in exclude_indices]
        
        if not project_indices:
            return None, None, None, None
            
        # Start from current project
        current_idx = self.get_current_project_index()
        
        # Reorder to start from current project
        if current_idx in project_indices:
            start_pos = project_indices.index(current_idx)
            project_indices = project_indices[start_pos:] + project_indices[:start_pos]
        
        for project_index in project_indices:
            router = self.project_routers[project_index]
            region, bucket = router.pick_region(tokens_needed)
            
            if region is not None:
                self.set_current_project_index(project_index)
                return project_index, self.project_names[project_index], region, bucket
                
        return None, None, None, None
    
    def pick_region_with_fallback(self, tokens_needed=1000, max_project_attempts=None):
        """
        Pick a region with automatic project fallback.
        Tries current project first, then falls back to other projects.
        """
        if max_project_attempts is None:
            max_project_attempts = len(self.project_routers)
            
        tried_projects = set()
        attempts = 0
        
        while attempts < max_project_attempts and len(tried_projects) < len(self.project_routers):
            project_index, project_name, region, bucket = self.get_next_available_project(
                tokens_needed, exclude_indices=tried_projects
            )
            
            if project_index is not None:
                return {
                    'project_index': project_index,
                    'project_name': project_name,
                    'region': region,
                    'bucket': bucket,
                    'tokens_allocated': tokens_needed
                }
            
            # Mark this project as tried
            current_idx = self.get_current_project_index()
            tried_projects.add(current_idx)
            attempts += 1
        
        return None
    
    def refund_tokens_to_project(self, project_index, region, tokens):
        """Refund tokens to a specific project and region"""
        if project_index not in self.project_routers:
            return False
            
        return self.project_routers[project_index].refund_tokens(region, tokens)
    
    def get_project_balances(self, project_identifier=None):
        """Get balances for a specific project or all projects"""
        if project_identifier is not None:
            # Handle both index and name
            if isinstance(project_identifier, str):
                if project_identifier in self.project_indices:
                    project_index = self.project_indices[project_identifier]
                else:
                    return {}
            else:
                project_index = project_identifier
                
            if project_index not in self.project_routers:
                return {}
                
            return {
                self.project_names[project_index]: self.project_routers[project_index].get_all_balances()
            }
        else:
            # Return all project balances
            all_balances = {}
            for project_index, router in self.project_routers.items():
                project_name = self.project_names[project_index]
                all_balances[project_name] = router.get_all_balances()
            return all_balances
    
    def get_project_info(self):
        """Get information about all projects"""
        return {
            'current_project_index': self.get_current_project_index(),
            'projects': {
                index: {
                    'name': name,
                    'regions': list(self.project_routers[index].region_list)
                }
                for index, name in self.project_names.items()
            }
        }
    
    def load_projects_from_storage(self):
        """Load projects from persistent storage (Redis or local)"""
        if self.use_redis:
            projects_data = self.redis_client.hgetall(self.projects_key)
            for index_str, name in projects_data.items():
                project_index = int(index_str)
                if project_index not in self.project_routers:
                    self.add_project(project_index, name)
        else:
            for project_index, name in self.projects_data.items():
                if project_index not in self.project_routers:
                    self.add_project(project_index, name)
    
    def close(self):
        """Close all project routers and Redis connection"""
        for router in self.project_routers.values():
            router.close()
            
        if self.use_redis and self.redis_client:
            self.redis_client.close()


