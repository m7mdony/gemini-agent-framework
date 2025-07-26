"""
Example usage of the Vertex AI Rotation Manager.
This demonstrates how to set up and use the rotation system with multiple projects.
"""
import os
import json
import time
from typing import List, Dict, Any
from vertex_agent.vertex_routing.router import VertexRotationManager


def create_sample_projects_config() -> List[Dict[str, Any]]:
    """Create sample project configurations."""
    # This is a template - replace with your actual project configurations
    projects = []
    
    # Example: 30 projects configuration
    for i in range(1, 31):  # 30 projects
        project_config = {
            "project_id": f"my-vertex-project-{i:02d}",
            "key_path": f"/path/to/service-account-keys/project-{i:02d}-key.json",
            "regions": [
                "us-central1", "us-east1", "us-east4", "us-west1", "us-west2",
                "us-west3", "us-west4", "europe-west1", "europe-west2", 
                "europe-west3", "europe-west4", "europe-west6", 
                "asia-east1", "asia-northeast1"  # 14 regions per project
            ]
        }
        projects.append(project_config)
    
    return projects


def create_your_projects_config() -> List[Dict[str, Any]]:
    """
    Create your actual project configurations.
    REPLACE THIS FUNCTION WITH YOUR REAL PROJECT DETAILS.
    """
    projects = [
        # Example project 1
        {
            "project_id": "your-actual-project-id-1",
            "key_path": "/path/to/your/service-account-key-1.json",
            "regions": [
                "us-central1", "us-east1", "us-east4", "us-west1", "us-west2",
                "us-west3", "us-west4", "europe-west1", "europe-west2", 
                "europe-west3", "europe-west4", "europe-west6", 
                "asia-east1", "asia-northeast1"
            ]
        },
        # Example project 2
        {
            "project_id": "your-actual-project-id-2", 
            "key_path": "/path/to/your/service-account-key-2.json",
            "regions": [
                "us-central1", "us-east1", "us-east4", "us-west1", "us-west2",
                "us-west3", "us-west4", "europe-west1", "europe-west2", 
                "europe-west3", "europe-west4", "europe-west6", 
                "asia-east1", "asia-northeast1"
            ]
        },
        # Add more projects here...
        # Copy the pattern above for each of your 30 projects
    ]
    
    return projects


def load_projects_from_config_file(config_file: str = "my_projects.json") -> List[Dict[str, Any]]:
    """
    Load project configurations from a JSON file.
    This is the recommended approach for managing many projects.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config.get("projects", [])
    except FileNotFoundError:
        print(f"Config file {config_file} not found. Creating template...")
        create_config_file_template(config_file)
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        return []


def create_config_file_template(filename: str = "my_projects.json"):
    """Create a template configuration file for your projects."""
    template = {
        "projects": [
            {
                "project_id": "your-project-id-1",
                "key_path": "/path/to/your/service-account-key-1.json",
                "regions": [
                    "us-central1", "us-east1", "us-east4", "us-west1", "us-west2",
                    "us-west3", "us-west4", "europe-west1", "europe-west2", 
                    "europe-west3", "europe-west4", "europe-west6", 
                    "asia-east1", "asia-northeast1"
                ],
                "enabled": True,
                "description": "Project 1 description"
            },
            {
                "project_id": "your-project-id-2",
                "key_path": "/path/to/your/service-account-key-2.json",
                "regions": [
                    "us-central1", "us-east1", "us-east4", "us-west1", "us-west2",
                    "us-west3", "us-west4", "europe-west1", "europe-west2", 
                    "europe-west3", "europe-west4", "europe-west6", 
                    "asia-east1", "asia-northeast1"
                ],
                "enabled": True,
                "description": "Project 2 description"
            }
            # Add more projects here...
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"📝 Template config file created: {filename}")
    print("✏️  Edit this file with your actual project details!")


def interactive_project_setup() -> List[Dict[str, Any]]:
    """Interactive setup for entering project configurations."""
    projects = []
    
    print("🔧 Interactive Project Setup")
    print("=" * 40)
    
    while True:
        print(f"\nSetting up project {len(projects) + 1}")
        
        project_id = input("Enter project ID: ").strip()
        if not project_id:
            break
            
        key_path = input("Enter path to service account key file: ").strip()
        if not key_path:
            break
            
        # Validate key file exists
        if not os.path.exists(key_path):
            print(f"⚠️  Warning: Key file not found at {key_path}")
            continue_anyway = input("Continue anyway? (y/n): ").strip().lower()
            if continue_anyway != 'y':
                continue
        
        # Use default regions or custom
        use_default_regions = input("Use default 14 regions? (y/n): ").strip().lower()
        if use_default_regions == 'y':
            regions = [
                "us-central1", "us-east1", "us-east4", "us-west1", "us-west2",
                "us-west3", "us-west4", "europe-west1", "europe-west2", 
                "europe-west3", "europe-west4", "europe-west6", 
                "asia-east1", "asia-northeast1"
            ]
        else:
            regions_input = input("Enter regions (comma-separated): ").strip()
            regions = [r.strip() for r in regions_input.split(',') if r.strip()]
        
        project_config = {
            "project_id": project_id,
            "key_path": key_path,
            "regions": regions,
            "enabled": True
        }
        
        projects.append(project_config)
        print(f"✅ Added project: {project_id}")
        
        # Ask if they want to add more
        if len(projects) >= 30:
            print("📊 Reached maximum of 30 projects!")
            break
            
        add_more = input("Add another project? (y/n): ").strip().lower()
        if add_more != 'y':
            break
    
    # Save to file
    if projects:
        save_to_file = input("Save configuration to file? (y/n): ").strip().lower()
        if save_to_file == 'y':
            filename = input("Enter filename (default: my_projects.json): ").strip()
            if not filename:
                filename = "my_projects.json"
            
            config = {"projects": projects}
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"💾 Configuration saved to {filename}")
    
    return projects


def main():
    """Main example function."""
    print("🚀 Vertex AI Rotation Manager - Setup Options")
    print("=" * 50)
    print("1. Use sample/template configuration")
    print("2. Use your actual project configuration")
    print("3. Load from JSON config file")
    print("4. Interactive setup")
    
    choice = input("\nChoose setup method (1-4): ").strip()
    
    if choice == "1":
        print("📋 Using sample configuration (for testing only)")
        projects_config = create_sample_projects_config()
    elif choice == "2":
        print("🔧 Using your actual project configuration")
        projects_config = create_your_projects_config()
    elif choice == "3":
        config_file = input("Enter config file path (default: my_projects.json): ").strip()
        if not config_file:
            config_file = "my_projects.json"
        projects_config = load_projects_from_config_file(config_file)
        if not projects_config:
            print("❌ No projects loaded. Exiting.")
            return
    elif choice == "4":
        print("🎯 Interactive project setup")
        projects_config = interactive_project_setup()
        if not projects_config:
            print("❌ No projects configured. Exiting.")
            return
    else:
        print("❌ Invalid choice. Using sample configuration.")
        projects_config = create_sample_projects_config()
    
    # Validate that we have projects
    if not projects_config:
        print("❌ No projects configured. Please check your configuration.")
        return
    
    print(f"\n✅ Loaded {len(projects_config)} projects")
    
    # Show first few projects for confirmation
    print("\n📊 Project Summary:")
    for i, project in enumerate(projects_config[:3]):
        key_status = "✅" if os.path.exists(project['key_path']) else "❌"
        print(f"  {i+1}. {project['project_id']}: {len(project['regions'])} regions {key_status}")
    
    if len(projects_config) > 3:
        print(f"  ... and {len(projects_config) - 3} more projects")
    
    # Ask for confirmation before proceeding
    confirm = input("\nProceed with API testing? (y/n): ").strip().lower()
    if confirm != 'y':
        print("👋 Exiting without testing.")
        return
    
    # Initialize rotation manager
    try:
        rotation_manager = VertexRotationManager(
            projects_config=projects_config,
            model_name="gemini-2.0-flash",
            rotation_strategy="least_used",  # Options: round_robin, least_used, random
            persistence_file="rotation_state.json"
        )
        print("✅ Rotation manager initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing rotation manager: {e}")
        return
    
    # Example API payloads
    sample_payloads = [
        {
            "contents": [{
                "role": "user",
                "parts": [{"text": "Hello! How are you today?"}]
            }]
        },
        {
            "contents": [{
                "role": "user", 
                "parts": [{"text": "Explain quantum computing in simple terms."}]
            }]
        },
        {
            "contents": [{
                "role": "user",
                "parts": [{"text": "Write a short story about a robot learning to paint."}]
            }]
        }
    ]
    
    # Make API calls with automatic rotation
    print("Starting API calls with automatic rotation...")
    
    for i, payload in enumerate(sample_payloads):
        try:
            print(f"\nMaking API call {i+1}...")
            response = rotation_manager.call_gemini_api(payload)
            
            # Extract response text
            if 'candidates' in response and response['candidates']:
                content = response['candidates'][0]['content']['parts'][0]['text']
                print(f"Response preview: {content[:100]}...")
            
            # Show current status
            status = rotation_manager.get_status()
            print(f"Current project: {status['current_project']}")
            print(f"Current region: {status['current_region']}")
            print(f"Success rate: {status['metrics']['success_rate']:.1f}%")
            
        except Exception as e:
            print(f"Error in API call {i+1}: {e}")
        
        # Small delay between calls
        time.sleep(1)
    
    # Print final status
    print("\n" + "="*50)
    print("FINAL ROTATION STATUS")
    print("="*50)
    
    final_status = rotation_manager.get_status()
    print(f"Total requests: {final_status['metrics']['total_requests']}")
    print(f"Successful requests: {final_status['metrics']['successful_requests']}")
    print(f"Failed requests: {final_status['metrics']['failed_requests']}")
    print(f"Success rate: {final_status['metrics']['success_rate']:.1f}%")
    print(f"Project switches: {final_status['metrics']['project_switches']}")
    print(f"Region switches: {final_status['metrics']['region_switches']}")
    print(f"Quota exceeded events: {final_status['metrics']['quota_exceeded_count']}")
    print(f"Rate limit hits: {final_status['metrics']['rate_limit_hits']}")
    
    # Show project status
    print("\nPROJECT STATUS:")
    for project in final_status['projects_status']:
        status_icon = "🔴" if project['is_blocked'] else "🟢"
        print(f"{status_icon} {project['project_id']}: {project['token_count']} tokens used")


def stress_test_rotation():
    """Stress test the rotation system with many rapid calls."""
    print("Starting stress test...")
    
    projects_config = create_sample_projects_config()
    
    with VertexRotationManager(
        projects_config=projects_config,
        model_name="gemini-1.5-flash",
        rotation_strategy="round_robin",
        persistence_file="stress_test_state.json"
    ) as rotation_manager:
        
        # Simple payload for stress testing
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": "Say hello!"}]
            }]
        }
        
        # Make many rapid calls
        for i in range(100):
            try:
                response = rotation_manager.call_gemini_api(payload)
                if i % 10 == 0:
                    status = rotation_manager.get_status()
                    print(f"Call {i+1}: Project {status['current_project']}, "
                          f"Region {status['current_region']}, "
                          f"Success Rate: {status['metrics']['success_rate']:.1f}%")
            except Exception as e:
                print(f"Call {i+1} failed: {e}")
            
            # Very small delay
            time.sleep(0.1)
        
        # Final status
        final_status = rotation_manager.get_status()
        print(f"\nStress test completed:")
        print(f"Total calls: {final_status['metrics']['total_requests']}")
        print(f"Success rate: {final_status['metrics']['success_rate']:.1f}%")


def monitor_rotation_health():
    """Monitor the health of the rotation system."""
    projects_config = create_sample_projects_config()
    
    rotation_manager = VertexRotationManager(
        projects_config=projects_config,
        model_name="gemini-2.0-flash",
        rotation_strategy="least_used",
        # persistence_file="health_monitor_state.json"
    )
    
    # Monitor for a period
    print("Monitoring rotation health...")
    
    for minute in range(5):  # Monitor for 5 minutes
        status = rotation_manager.get_status()
        
        # Calculate health metrics
        active_projects = status['active_projects']
        total_projects = status['total_projects']
        health_percentage = (active_projects / total_projects) * 100
        
        print(f"\nMinute {minute + 1}:")
        print(f"  Active projects: {active_projects}/{total_projects} ({health_percentage:.1f}%)")
        print(f"  Current: {status['current_project']} in {status['current_region']}")
        print(f"  Switches: {status['metrics']['project_switches']} projects, "
              f"{status['metrics']['region_switches']} regions")
        
        # Alert if health is low
        if health_percentage < 50:
            print("  ⚠️  WARNING: Less than 50% of projects are active!")
        
        # Make a test call
        test_payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": f"Health check {minute + 1}"}]
            }]
        }
        
        try:
            rotation_manager.call_gemini_api(test_payload)
            print("  ✅ Test call successful")
        except Exception as e:
            print(f"  ❌ Test call failed: {e}")
        
        # Wait for next minute
        time.sleep(60)


if __name__ == "__main__":
    # Run basic example
    main()
    
    # Uncomment to run stress test
    # stress_test_rotation()
    
    # Uncomment to run health monitoring
    # monitor_rotation_health()