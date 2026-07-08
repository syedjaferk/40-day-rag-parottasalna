from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# DATASET

documents = [
    # Programming
    "Python variables store values. Data types include integers, floats, strings, lists, tuples, dictionaries, and sets.",
    "Python functions help organize reusable code. Functions can accept parameters and return values.",
    "Object-Oriented Programming in Python introduces classes, objects, inheritance, encapsulation, polymorphism, and abstraction.",
    "Python decorators extend the functionality of existing functions without modifying their source code.",
    "Generators produce values lazily using the yield keyword, making them memory efficient.",
    "Exception handling in Python uses try, except, else, finally, and custom exceptions.",
    # Database
    "SQL SELECT retrieves records from tables. WHERE filters rows, and ORDER BY sorts the results.",
    "Primary keys uniquely identify records, while foreign keys maintain relationships between tables.",
    "Indexes improve database performance by reducing the number of scanned rows.",
    "Database normalization reduces redundancy using First, Second, and Third Normal Forms.",
    "Transactions guarantee ACID properties including Atomicity, Consistency, Isolation, and Durability.",
    "MongoDB stores data as BSON documents inside collections instead of relational tables.",
    # DevOps
    "Docker packages applications into lightweight containers using images and isolated runtimes.",
    "Docker Compose manages multi-container applications using YAML configuration files.",
    "Kubernetes Pods are the smallest deployable units and run one or more containers.",
    "Deployments manage application updates and rolling deployments inside Kubernetes.",
    "Services expose Kubernetes Pods using ClusterIP, NodePort, and LoadBalancer.",
    "Helm simplifies Kubernetes application deployment using reusable charts.",
    # AI
    "Machine Learning enables systems to learn patterns from historical data.",
    "Supervised learning uses labeled datasets for prediction tasks like classification and regression.",
    "Unsupervised learning discovers hidden structures using clustering and dimensionality reduction.",
    "Neural Networks consist of interconnected layers of neurons trained using backpropagation.",
    "Large Language Models generate text using Transformer architectures and attention mechanisms.",
    "Retrieval Augmented Generation combines vector search with language models to improve factual accuracy.",
    # Cloud
    "Amazon EC2 provides virtual machines in the cloud with different instance types.",
    "Amazon S3 stores objects with high durability and supports lifecycle management.",
    "AWS IAM manages authentication, authorization, users, groups, roles, and policies.",
    "AWS Lambda executes serverless functions without provisioning servers.",
    # Networking
    "HTTP is a stateless protocol used for communication between clients and servers.",
    "Load balancers distribute incoming traffic across multiple backend servers to improve availability.",
    # Security
    "Encryption converts plaintext into ciphertext using symmetric and asymmetric algorithms.",
    "OAuth 2.0 provides delegated authorization for web and mobile applications.",
]

metadatas = [
    {
        "category": "Programming",
        "topic": "Python",
        "author": "Eric Matthes",
        "level": "Beginner",
        "year": 2023,
    },
    {
        "category": "Programming",
        "topic": "Python",
        "author": "Eric Matthes",
        "level": "Beginner",
        "year": 2023,
    },
    {
        "category": "Programming",
        "topic": "Python OOP",
        "author": "Luciano Ramalho",
        "level": "Intermediate",
        "year": 2022,
    },
    {
        "category": "Programming",
        "topic": "Decorators",
        "author": "Luciano Ramalho",
        "level": "Advanced",
        "year": 2022,
    },
    {
        "category": "Programming",
        "topic": "Generators",
        "author": "Luciano Ramalho",
        "level": "Advanced",
        "year": 2022,
    },
    {
        "category": "Programming",
        "topic": "Exception Handling",
        "author": "Eric Matthes",
        "level": "Intermediate",
        "year": 2023,
    },
    {
        "category": "Database",
        "topic": "SQL",
        "author": "Alan Beaulieu",
        "level": "Beginner",
        "year": 2021,
    },
    {
        "category": "Database",
        "topic": "Keys",
        "author": "Alan Beaulieu",
        "level": "Beginner",
        "year": 2021,
    },
    {
        "category": "Database",
        "topic": "Indexing",
        "author": "Mark White",
        "level": "Intermediate",
        "year": 2024,
    },
    {
        "category": "Database",
        "topic": "Normalization",
        "author": "Mark White",
        "level": "Intermediate",
        "year": 2024,
    },
    {
        "category": "Database",
        "topic": "Transactions",
        "author": "Mark White",
        "level": "Advanced",
        "year": 2024,
    },
    {
        "category": "Database",
        "topic": "MongoDB",
        "author": "Kristina Chodorow",
        "level": "Intermediate",
        "year": 2023,
    },
    {
        "category": "DevOps",
        "topic": "Docker",
        "author": "Nigel Poulton",
        "level": "Beginner",
        "year": 2024,
    },
    {
        "category": "DevOps",
        "topic": "Docker Compose",
        "author": "Nigel Poulton",
        "level": "Intermediate",
        "year": 2024,
    },
    {
        "category": "DevOps",
        "topic": "Kubernetes",
        "author": "Marko Luksa",
        "level": "Intermediate",
        "year": 2023,
    },
    {
        "category": "DevOps",
        "topic": "Deployments",
        "author": "Marko Luksa",
        "level": "Advanced",
        "year": 2023,
    },
    {
        "category": "DevOps",
        "topic": "Services",
        "author": "Marko Luksa",
        "level": "Intermediate",
        "year": 2023,
    },
    {
        "category": "DevOps",
        "topic": "Helm",
        "author": "Matt Butcher",
        "level": "Advanced",
        "year": 2024,
    },
    {
        "category": "Artificial Intelligence",
        "topic": "Machine Learning",
        "author": "Aurélien Géron",
        "level": "Beginner",
        "year": 2022,
    },
    {
        "category": "Artificial Intelligence",
        "topic": "Supervised Learning",
        "author": "Aurélien Géron",
        "level": "Intermediate",
        "year": 2022,
    },
    {
        "category": "Artificial Intelligence",
        "topic": "Unsupervised Learning",
        "author": "Aurélien Géron",
        "level": "Intermediate",
        "year": 2022,
    },
    {
        "category": "Artificial Intelligence",
        "topic": "Neural Networks",
        "author": "François Chollet",
        "level": "Advanced",
        "year": 2024,
    },
    {
        "category": "Artificial Intelligence",
        "topic": "LLM",
        "author": "Sebastian Raschka",
        "level": "Advanced",
        "year": 2025,
    },
    {
        "category": "Artificial Intelligence",
        "topic": "RAG",
        "author": "Sebastian Raschka",
        "level": "Advanced",
        "year": 2025,
    },
    {
        "category": "Cloud",
        "topic": "EC2",
        "author": "AWS",
        "level": "Beginner",
        "year": 2025,
    },
    {
        "category": "Cloud",
        "topic": "S3",
        "author": "AWS",
        "level": "Beginner",
        "year": 2025,
    },
    {
        "category": "Cloud",
        "topic": "IAM",
        "author": "AWS",
        "level": "Intermediate",
        "year": 2025,
    },
    {
        "category": "Cloud",
        "topic": "Lambda",
        "author": "AWS",
        "level": "Intermediate",
        "year": 2025,
    },
    {
        "category": "Networking",
        "topic": "HTTP",
        "author": "MDN",
        "level": "Beginner",
        "year": 2024,
    },
    {
        "category": "Networking",
        "topic": "Load Balancer",
        "author": "NGINX",
        "level": "Intermediate",
        "year": 2024,
    },
    {
        "category": "Cyber Security",
        "topic": "Encryption",
        "author": "Bruce Schneier",
        "level": "Intermediate",
        "year": 2023,
    },
    {
        "category": "Cyber Security",
        "topic": "OAuth",
        "author": "OAuth Working Group",
        "level": "Advanced",
        "year": 2024,
    },
]

ids = [f"doc_{i}" for i in range(1, len(documents) + 1)]

# CREATE VECTOR DATABASE

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    metadatas=metadatas,
    ids=ids,
    persist_directory="./chroma_db",
)

db.persist()

print("Vector Database Created Successfully.")

# SEARCH FUNCTION


def search(query, metadata_filter=None, k=3):
    docs = db.similarity_search(query=query, k=k, filter=metadata_filter)

    print("\n" + "=" * 80)
    print("Query :", query)
    print("Filter:", metadata_filter)
    print("=" * 80)

    if not docs:
        print("No Results Found")
        return

    for i, doc in enumerate(docs, start=1):
        print(f"\nResult {i}")
        print("-" * 80)

        print("Metadata")

        for key, value in doc.metadata.items():
            print(f"{key:12}: {value}")

        print("\nContent")
        print(doc.page_content)


while True:
    print("\n")

    query = input("Question (q to quit): ")

    if query.lower() == "q":
        break


    filters = {}

    if category:
        filters["category"] = category

    if author:
        filters["author"] = author

    if level:
        filters["level"] = level

    search(query=query, metadata_filter=filters if filters else None)
