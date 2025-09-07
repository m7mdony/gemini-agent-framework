import threading
import time
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from vertex_agent import Agent
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import requests

# Token counting utility using Vertex AI REST API
def count_tokens_vertex(text, project_id, key_path, model_name="gemini-2.0-flash", location="us-central1"):
    """Count tokens using Vertex AI REST API with service account authentication"""
    try:
        # Load service account credentials
        credentials = service_account.Credentials.from_service_account_file(
            key_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        
        # Refresh credentials to get access token
        auth_req = Request()
        credentials.refresh(auth_req)
        
        # Prepare the API endpoint
        url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_name}:countTokens"
        
        # Prepare the request payload
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": text}]
                }
            ]
        }
        
        # Make the API request
        headers = {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("totalTokens", 0)
        else:
            print(f"Warning: Token counting API returned {response.status_code}: {response.text}")
            # Fallback to rough estimation
            return len(text) // 4
            
    except Exception as e:
        print(f"Warning: Could not count tokens using Vertex AI: {e}")
        # Fallback to rough estimation
        return len(text) // 4


# Generate a large text prompt (50k-150k tokens)
def generate_large_prompt():
    # Base text that will be repeated to reach token count
    base_text = """
    This is a comprehensive analysis of artificial intelligence systems, machine learning algorithms, 
    natural language processing techniques, computer vision methodologies, robotics applications, 
    data science practices, software engineering principles, cloud computing architectures, 
    cybersecurity protocols, database management systems, network infrastructure design, 
    mobile application development, web technologies, user experience design, project management, 
    business intelligence, digital transformation strategies, innovation frameworks, 
    technological advancement patterns, research methodologies, academic publications, 
    industry best practices, regulatory compliance requirements, ethical considerations, 
    sustainability initiatives, global market trends, economic implications, social impacts, 
    cultural influences, educational approaches, training methodologies, skill development, 
    career progression paths, professional certifications, continuous learning, 
    knowledge management, information systems, decision support tools, analytics platforms, 
    visualization techniques, reporting mechanisms, performance metrics, quality assurance, 
    testing frameworks, deployment strategies, monitoring solutions, maintenance procedures, 
    troubleshooting guides, documentation standards, collaboration tools, communication protocols, 
    stakeholder engagement, customer satisfaction, service delivery, operational efficiency, 
    cost optimization, resource allocation, risk management, change management, governance models, 
    compliance frameworks, audit procedures, security measures, data protection, privacy policies, 
    access controls, authentication mechanisms, authorization systems, encryption technologies, 
    backup strategies, disaster recovery plans, business continuity, incident response, 
    vulnerability assessments, threat intelligence, security awareness, training programs, 
    policy development, procedure implementation, standard operating procedures, 
    workflow optimization, process improvement, automation opportunities, digital workflows, 
    integration capabilities, interoperability standards, API development, microservices architecture, 
    containerization technologies, orchestration platforms, DevOps practices, CI/CD pipelines, 
    version control systems, code review processes, testing automation, deployment automation, 
    infrastructure as code, configuration management, monitoring and alerting, log management, 
    performance optimization, scalability considerations, load balancing, caching strategies, 
    content delivery networks, edge computing, distributed systems, fault tolerance, 
    high availability, disaster recovery, backup and restore, data replication, 
    consistency models, transaction processing, concurrency control, database optimization, 
    query performance, indexing strategies, data modeling, schema design, normalization, 
    denormalization, data warehousing, ETL processes, data pipelines, streaming data, 
    real-time processing, batch processing, data lakes, data governance, data quality, 
    master data management, metadata management, data lineage, data catalog, 
    data discovery, data privacy, data security, data retention, data archiving, 
    data migration, data integration, data transformation, data validation, data cleansing, 
    data profiling, data analysis, statistical analysis, predictive modeling, 
    machine learning algorithms, deep learning networks, neural networks, 
    artificial intelligence applications, natural language processing, computer vision, 
    speech recognition, image processing, pattern recognition, anomaly detection, 
    recommendation systems, classification algorithms, clustering techniques, 
    regression analysis, time series analysis, forecasting models, optimization algorithms, 
    genetic algorithms, evolutionary computation, swarm intelligence, fuzzy logic, 
    expert systems, knowledge representation, reasoning systems, inference engines, 
    ontologies, semantic web, linked data, knowledge graphs, information retrieval, 
    search algorithms, ranking algorithms, relevance scoring, personalization, 
    collaborative filtering, content-based filtering, hybrid approaches, 
    user modeling, behavior analysis, sentiment analysis, opinion mining, 
    text mining, web scraping, data extraction, information extraction, 
    named entity recognition, part-of-speech tagging, syntactic parsing, 
    semantic parsing, discourse analysis, pragmatics, computational linguistics, 
    corpus linguistics, language models, n-gram models, hidden Markov models, 
    conditional random fields, maximum entropy models, support vector machines, 
    decision trees, random forests, gradient boosting, ensemble methods, 
    cross-validation, model selection, hyperparameter tuning, feature engineering, 
    feature selection, dimensionality reduction, principal component analysis, 
    independent component analysis, linear discriminant analysis, 
    manifold learning, clustering algorithms, k-means clustering, 
    hierarchical clustering, density-based clustering, spectral clustering, 
    mixture models, expectation-maximization, variational inference, 
    Markov chain Monte Carlo, Bayesian networks, probabilistic graphical models, 
    causal inference, experimental design, A/B testing, statistical significance, 
    hypothesis testing, confidence intervals, p-values, effect sizes, 
    power analysis, sample size determination, survey design, data collection, 
    observational studies, longitudinal studies, cross-sectional studies, 
    case-control studies, cohort studies, randomized controlled trials, 
    systematic reviews, meta-analysis, evidence-based practice, 
    research methodology, scientific method, peer review, publication ethics, 
    reproducibility, replicability, open science, data sharing, 
    collaboration, interdisciplinary research, translational research, 
    innovation ecosystems, technology transfer, commercialization, 
    intellectual property, patents, licensing, startups, entrepreneurship, 
    venture capital, funding mechanisms, business models, value propositions, 
    market analysis, competitive analysis, SWOT analysis, strategic planning, 
    roadmapping, portfolio management, product management, project management, 
    agile methodologies, scrum framework, kanban boards, lean principles, 
    six sigma, total quality management, continuous improvement, 
    kaizen, change management, organizational behavior, leadership, 
    team dynamics, communication skills, negotiation, conflict resolution, 
    emotional intelligence, cultural competence, diversity and inclusion, 
    work-life balance, professional development, career planning, 
    networking, mentoring, coaching, performance management, 
    talent acquisition, recruitment, onboarding, training and development, 
    competency models, succession planning, retention strategies, 
    employee engagement, motivation theories, job satisfaction, 
    organizational culture, values, mission, vision, strategic objectives, 
    key performance indicators, balanced scorecard, dashboard reporting, 
    business intelligence, competitive intelligence, market research, 
    customer insights, user research, persona development, journey mapping, 
    service design, experience design, interaction design, visual design, 
    information architecture, usability testing, accessibility, 
    universal design, inclusive design, design thinking, human-centered design, 
    co-creation, participatory design, iterative design, prototyping, 
    wireframing, mockups, design systems, style guides, brand guidelines, 
    typography, color theory, layout principles, grid systems, 
    responsive design, mobile-first design, progressive enhancement, 
    graceful degradation, cross-browser compatibility, web standards, 
    semantic HTML, CSS methodologies, JavaScript frameworks, 
    front-end development, back-end development, full-stack development, 
    database design, API design, security best practices, 
    performance optimization, scalability patterns, architecture patterns, 
    design patterns, software engineering principles, code quality, 
    code review, refactoring, technical debt, legacy systems, 
    modernization strategies, migration planning, risk assessment, 
    mitigation strategies, contingency planning, crisis management, 
    business continuity planning, disaster recovery procedures, 
    incident management, problem management, change management, 
    release management, configuration management, service management, 
    IT service management, ITIL framework, service level agreements, 
    key performance indicators, service metrics, customer satisfaction, 
    continuous service improvement, process optimization, workflow automation, 
    digital transformation, cloud migration, hybrid cloud, multi-cloud, 
    edge computing, Internet of Things, artificial intelligence, 
    machine learning, blockchain technology, quantum computing, 
    augmented reality, virtual reality, mixed reality, 
    extended reality, metaverse, digital twins, smart cities, 
    Industry 4.0, digital manufacturing, supply chain optimization, 
    logistics management, inventory management, demand forecasting, 
    procurement processes, vendor management, contract management, 
    compliance monitoring, regulatory reporting, audit trails, 
    governance frameworks, risk management, internal controls, 
    fraud detection, cybersecurity, information security, 
    data protection, privacy engineering, consent management, 
    identity and access management, privileged access management, 
    single sign-on, multi-factor authentication, biometric authentication, 
    zero trust architecture, security by design, threat modeling, 
    vulnerability management, penetration testing, security assessment, 
    incident response, forensics, malware analysis, threat hunting, 
    security operations center, security information and event management, 
    security orchestration, automation and response, threat intelligence, 
    cyber threat landscape, attack vectors, social engineering, 
    phishing, ransomware, advanced persistent threats, insider threats, 
    supply chain attacks, IoT security, cloud security, mobile security, 
    application security, network security, endpoint security, 
    email security, web security, database security, encryption, 
    key management, digital certificates, public key infrastructure, 
    secure coding practices, security testing, static analysis, 
    dynamic analysis, interactive application security testing, 
    software composition analysis, container security, 
    Kubernetes security, DevSecOps, security automation, 
    compliance automation, continuous monitoring, security metrics, 
    security dashboards, executive reporting, board reporting, 
    stakeholder communication, security awareness training, 
    phishing simulation, security culture, behavioral security, 
    human factors in security, usable security, security psychology, 
    security economics, return on security investment, 
    cost-benefit analysis, risk quantification, actuarial models, 
    insurance models, cyber insurance, security standards, 
    certification programs, professional development, 
    security communities, information sharing, collaboration, 
    public-private partnerships, international cooperation, 
    cybersecurity policy, regulation, legislation, enforcement, 
    prosecution, attribution, deterrence, resilience, recovery, 
    lessons learned, best practices, emerging threats, 
    future trends, research directions, innovation opportunities, 
    technology roadmaps, strategic planning, investment priorities, 
    resource allocation, capability development, skill gaps, 
    workforce development, education and training, 
    curriculum development, certification programs, 
    professional associations, conferences, workshops, 
    publications, research papers, case studies, 
    white papers, technical reports, standards documents, 
    guidelines, frameworks, methodologies, tools, 
    technologies, platforms, solutions, services, 
    products, vendors, suppliers, partners, customers, 
    users, stakeholders, communities, ecosystems, 
    markets, industries, sectors, domains, applications, 
    use cases, scenarios, requirements, specifications, 
    designs, implementations, deployments, operations, 
    maintenance, support, documentation, training, 
    knowledge transfer, transition planning, 
    project closure, lessons learned, continuous improvement.
    """
    
    # Calculate approximate tokens (rough estimate: 1 token ≈ 4 characters)
    target_tokens = random.randint(50000, 150000)
    target_chars = target_tokens * 4
    
    # Repeat base text to reach target character count
    repetitions = max(1, target_chars // len(base_text))
    large_text = base_text * repetitions
    
    return f"Please analyze the following comprehensive text and respond with exactly one word that best summarizes the main theme: {large_text}"

def execute_agent_request(agent, request_id):
    """Execute a single agent request"""
    start_time = time.time()
    
    try:
        # Generate large prompt
        user_prompt = generate_large_prompt()
        
        # System prompt to ensure single word response
        system_prompt = """You are a helpful assistant. You must respond with exactly ONE WORD only. 
        No punctuation, no explanations, no additional text. Just one single word that best summarizes or represents the main theme of the input."""
        
        # Make the request
        response = agent.prompt(
            user_prompt=user_prompt,
            system_prompt=system_prompt
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Clean response to ensure it's one word
        if isinstance(response, str):
            clean_response = response.strip().split()[0] if response.strip() else "ERROR"
        elif isinstance(response, dict):
            clean_response = response.get("error", {}).get("message", "ERROR")
        else:
            clean_response = "ERROR"
            
        return {
            'request_id': request_id,
            'response': clean_response,
            'duration': duration,
            'success': True,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
            'tokens_approx': len(user_prompt) // 4  # Rough token estimate
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'request_id': request_id,
            'response': None,
            'duration': duration,
            'success': False,
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        }

def run_stress_test():
    """Run the stress test with concurrent requests"""
    
    # Configuration
    NUM_CONCURRENT_THREADS = 10  # Adjust based on your needs
    TOTAL_REQUESTS = 50  # Total number of requests to make
    
    # Router projects configuration
    router_projects = [
        {
            "project_id": "long-memory-465714-j2",
            "key_path": "/home/arete/capstone/1.json",
        },
        {
            "project_id": "browsemate1", 
            "key_path": "/home/arete/capstone/2.json",
        }
    ]

    # Initialize agent
    agent = Agent(
        model_name="gemini-2.0-flash",
        use_router=True,
        router_projects=router_projects,
        router_debug_mode=True
    )
    
    print(f"Starting stress test...")
    print(f"Concurrent threads: {NUM_CONCURRENT_THREADS}")
    print(f"Total requests: {TOTAL_REQUESTS}")
    print(f"Each request: 50k-150k tokens")
    print("-" * 50)
    
    # Track results
    results = []
    start_time = time.time()
    
    # Execute requests concurrently
    with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_THREADS) as executor:
        # Submit all requests
        future_to_request = {
            executor.submit(execute_agent_request, agent, i): i 
            for i in range(1, TOTAL_REQUESTS + 1)
        }
        
        # Process completed requests
        for future in as_completed(future_to_request):
            result = future.result()
            results.append(result)
            
            # Print progress
            if result['success']:
                print(f"✓ Request {result['request_id']}: '{result['response']}' "
                      f"({result['duration']:.2f}s, ~{result['tokens_approx']:,} tokens)")
            else:
                print(f"✗ Request {result['request_id']}: FAILED - {result['error']} "
                      f"({result['duration']:.2f}s)")
    
    # Calculate and display statistics
    total_time = time.time() - start_time
    successful_requests = [r for r in results if r['success']]
    failed_requests = [r for r in results if not r['success']]
    
    print("\n" + "=" * 60)
    print("STRESS TEST RESULTS")
    print("=" * 60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total requests: {TOTAL_REQUESTS}")
    print(f"Successful requests: {len(successful_requests)}")
    print(f"Failed requests: {len(failed_requests)}")
    print(f"Success rate: {len(successful_requests)/TOTAL_REQUESTS*100:.1f}%")
    
    if successful_requests:
        durations = [r['duration'] for r in successful_requests]
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        print(f"\nTiming Statistics:")
        print(f"Average response time: {avg_duration:.2f}s")
        print(f"Fastest response: {min_duration:.2f}s")
        print(f"Slowest response: {max_duration:.2f}s")
        print(f"Requests per second: {len(successful_requests)/total_time:.2f}")
    
    if successful_requests:
        print(f"\nSample responses:")
        for i, result in enumerate(successful_requests[:10]):  # Show first 10
            print(f"  {result['request_id']}: '{result['response']}'")
        if len(successful_requests) > 10:
            print(f"  ... and {len(successful_requests) - 10} more")
    
    if failed_requests:
        print(f"\nError Summary:")
        error_counts = {}
        for result in failed_requests:
            error_type = result['error'][:50] + "..." if len(result['error']) > 50 else result['error']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        for error, count in error_counts.items():
            print(f"  {error}: {count} times")
    
    return results

if __name__ == "__main__":
    results = run_stress_test()