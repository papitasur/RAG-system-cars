import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import openai
import re
from dotenv import load_dotenv

# LangChain imports
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Load environment variables
load_dotenv()

class CarBrochureRAG:
    def __init__(self, openai_api_key: str = None):
        """Initialize the RAG system with OpenAI API key"""
        if openai_api_key is None:
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize LangChain components
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0.3, max_tokens=1000)
        self.vectorstore = None
        self.qa_chain = None
        self.documents = []
        self.car_data = {}
        
        # Text splitter for better chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def extract_table_data(self, text: str) -> Dict[str, Any]:
        """Enhanced table data extraction with better patterns"""
        # Patterns for Indian car specifications
        patterns = {
            'car_model': r'(?:Model|Car|Vehicle)[\s:]*([A-Za-z0-9\s]+?)(?:\n|Price|Engine)',
            'variant': r'(?:Variant|Grade|Trim)[\s:]*([A-Za-z0-9\s\-]+?)(?:\n|₹)',
            'price': r'₹\s*([\d,]+(?:\.\d+)?)\s*(?:lakh|Lakh|LAKH)?',
            'ex_showroom': r'(?:Ex-showroom|Ex showroom)[\s:]*₹\s*([\d,]+(?:\.\d+)?)',
            'engine_cc': r'(\d{3,4})\s*(?:cc|CC)',
            'engine_litre': r'(\d+\.\d+)\s*(?:L|l|litre|Litre)',
            'mileage_kmpl': r'(\d+(?:\.\d+)?)\s*(?:kmpl|KMPL|km/l|Km/L)',
            'power_bhp': r'(\d+(?:\.\d+)?)\s*(?:bhp|BHP|PS|ps)',
            'torque_nm': r'(\d+(?:\.\d+)?)\s*(?:Nm|NM|nm)',
            'fuel_type': r'\b(Petrol|Diesel|Electric|Hybrid|CNG|petrol|diesel|electric|hybrid|cng)\b',
            'transmission': r'\b(Manual|Automatic|AMT|CVT|manual|automatic|amt|cvt)\b',
            'seating': r'(\d+)\s*(?:seater|Seater|seats|Seats)',
            'fuel_tank': r'(\d+)\s*(?:litres?|L)\s*(?:fuel tank|tank)',
            'ground_clearance': r'(\d+)\s*(?:mm)\s*(?:ground clearance|clearance)',
            'boot_space': r'(\d+)\s*(?:litres?|L)\s*(?:boot|luggage)',
            'airbags': r'(\d+)\s*(?:airbags?|Airbags?)',
            'abs': r'\b(ABS|abs)\b',
            'ebd': r'\b(EBD|ebd)\b'
        }
        
        extracted_data = {}
        
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            if matches:
                # Clean and store matches
                cleaned_matches = [match.strip() for match in matches if match.strip()]
                extracted_data[key] = cleaned_matches
        
        return extracted_data
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents and extract structured data"""
        processed_docs = []
        
        for doc in documents:
            # Extract structured data from content
            extracted_data = self.extract_table_data(doc.page_content)
            
            # Enhance metadata with extracted information
            enhanced_metadata = doc.metadata.copy()
            enhanced_metadata.update(extracted_data)
            
            # Store car data for reference
            filename = enhanced_metadata.get('source', 'unknown')
            self.car_data[filename] = {
                'extracted_specs': extracted_data,
                'content_preview': doc.page_content[:500]
            }
            
            # Create enhanced document
            processed_doc = Document(
                page_content=doc.page_content,
                metadata=enhanced_metadata
            )
            processed_docs.append(processed_doc)
        
        return processed_docs
    
    def load_pdfs(self, pdf_folder: str):
        """Load and process all PDFs from a folder using LangChain"""
        print(f"Checking directory: {pdf_folder}")
        
        if not os.path.exists(pdf_folder):
            print(f"Folder {pdf_folder} does not exist!")
            return
        
        # Check for PDF files
        all_files = os.listdir(pdf_folder)
        pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
        
        print(f"Total files in directory: {len(all_files)}")
        print(f"PDF files found: {len(pdf_files)}")
        
        if pdf_files:
            print("PDF files:")
            for pdf in pdf_files:
                pdf_path = os.path.join(pdf_folder, pdf)
                size = os.path.getsize(pdf_path) / 1024  # Size in KB
                print(f"  • {pdf} ({size:.1f} KB)")
        
        if not pdf_files:
            print("No PDF files found in the folder!")
            print(f"Files in directory: {all_files}")
            return
        
        print(f"Found {len(pdf_files)} PDF files. Processing with LangChain...")
        
        try:
            # Load PDFs using LangChain DirectoryLoader
            loader = DirectoryLoader(
                pdf_folder,
                glob="*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from PDFs")
            
            # Debug: Check if documents have content
            if documents:
                print("Sample content from first document:")
                print(f"Content length: {len(documents[0].page_content)} characters")
                print(f"First 200 chars: {documents[0].page_content[:200]}...")
                print(f"Metadata: {documents[0].metadata}")
            else:
                print("No content extracted from PDFs!")
            
            # Process documents and extract structured data
            processed_docs = self.process_documents(documents)
            
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(processed_docs)
            print(f"Split into {len(split_docs)} chunks")
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            print("Created FAISS vector store")
            
            # Create custom prompt template
            custom_prompt = PromptTemplate(
                template="""You are a car expert assistant specializing in Indian car market data as of July 2025. 
                Use the following car brochure information to answer questions about car specifications, pricing, and comparisons.

                Context: {context}

                Question: {question}

                Instructions:
                - Focus on exact pricing, specifications, and technical details
                - Extract information from tables when available
                - List car variants with their specific details
                - Compare cars when asked, showing clear differences
                - Use specific numbers (prices in ₹ lakhs, mileage in kmpl, engine in cc/L, power in bhp/PS)
                - If information is in tabular form, present it clearly
                - Mention fuel type, transmission type, and key features

                Answer:""",
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                ),
                chain_type_kwargs={"prompt": custom_prompt},
                return_source_documents=True
            )
            
            print("RAG system initialized successfully!")
            
        except Exception as e:
            print(f"Error loading PDFs: {e}")
    
    def query(self, question: str) -> str:
        """Query the RAG system"""
        if not self.qa_chain:
            return "RAG system not initialized. Please load PDFs first."
        
        print(f"\nQuery: {question}")
        print("-" * 50)
        
        try:
            # Get response from QA chain
            result = self.qa_chain({"query": question})
            
            # Extract answer and sources
            answer = result["result"]
            source_docs = result.get("source_documents", [])
            
            # Add source information
            if source_docs:
                sources = []
                for doc in source_docs:
                    source = doc.metadata.get('source', 'Unknown source')
                    sources.append(os.path.basename(source))
                
                unique_sources = list(set(sources))
                answer += f"\n\nSources: {', '.join(unique_sources)}"
            
            return answer
            
        except Exception as e:
            return f"Error processing query: {e}"
    
    def list_loaded_cars(self):
        """List all loaded car brochures with extracted data"""
        print("\nLoaded Car Brochures:")
        print("-" * 50)
        
        if not self.car_data:
            print("No car data loaded yet.")
            return
        
        for filename, data in self.car_data.items():
            print(f"{os.path.basename(filename)}")
            
            specs = data.get('extracted_specs', {})
            if specs:
                print("   Extracted Information:")
                for key, values in specs.items():
                    if values:  # Only show non-empty values
                        print(f"   • {key.replace('_', ' ').title()}: {', '.join(map(str, values[:3]))}")
            
            print(f"   Preview: {data.get('content_preview', '')[:100]}...")
            print()
    
    def get_all_car_specs(self) -> str:
        """Get summary of all cars and their specifications"""
        if not self.car_data:
            return "No car data available."
        
        summary = "SUMMARY OF ALL LOADED CARS:\n" + "="*50 + "\n"
        
        for filename, data in self.car_data.items():
            car_name = os.path.basename(filename).replace('.pdf', '').replace('_', ' ').title()
            summary += f"\n{car_name}\n" + "-"*30 + "\n"
            
            specs = data.get('extracted_specs', {})
            key_specs = ['car_model', 'variant', 'price', 'engine_cc', 'mileage_kmpl', 'power_bhp', 'fuel_type', 'transmission']
            
            for spec in key_specs:
                if spec in specs and specs[spec]:
                    values = specs[spec][:2]  # Show first 2 values
                    summary += f"{spec.replace('_', ' ').title()}: {', '.join(map(str, values))}\n"
        
        return summary

# Usage Example
def main():
    print("CAR BROCHURE RAG SYSTEM (with LangChain)")
    print("="*60)
    
    try:
        # Initialize the RAG system (API key will be loaded from .env)
        rag_system = CarBrochureRAG()
        
        # Load PDFs from the specified directory
        pdf_folder = r"D:\Bhumin\cars pdf"
        rag_system.load_pdfs(pdf_folder)
        
        # List loaded cars
        rag_system.list_loaded_cars()
        
        # Sample queries optimized for tabular data
        sample_queries = [
            "Show me a comparison table of all car prices by variants",
            "What are the mileage figures for all cars?",
            "List all diesel variants with their specifications",
            "Which car has the highest power output?",
            "Compare the engine specifications of all cars",
            "Show me all automatic transmission options",
            "What are the top variant prices for each car?",
            "List all cars with their fuel types and transmission types"
        ]
        
        print("\n" + "="*60)
        print("LANGCHAIN RAG SYSTEM READY")
        print("="*60)
        
        # Interactive mode
        while True:
            print("\nOptions:")
            print("1. Ask a question")
            print("2. Try sample questions")
            print("3. List loaded cars")
            print("4. Get all car specs summary")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                question = input("\nEnter your question about the cars: ").strip()
                if question:
                    answer = rag_system.query(question)
                    print(f"\nAnswer:\n{answer}")
            
            elif choice == "2":
                print("\nSample Questions (optimized for table data):")
                for i, q in enumerate(sample_queries, 1):
                    print(f"{i}. {q}")
                
                try:
                    q_num = int(input("\nSelect question number: ")) - 1
                    if 0 <= q_num < len(sample_queries):
                        answer = rag_system.query(sample_queries[q_num])
                        print(f"\nAnswer:\n{answer}")
                    else:
                        print("Invalid question number!")
                except ValueError:
                    print("Please enter a valid number!")
            
            elif choice == "3":
                rag_system.list_loaded_cars()
            
            elif choice == "4":
                summary = rag_system.get_all_car_specs()
                print(f"\n{summary}")
            
            elif choice == "5":
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice! Please try again.")
    
    except Exception as e:
        print(f"Error initializing system: {e}")
        print("Make sure your .env file contains OPENAI_API_KEY and the PDF folder exists.")

if __name__ == "__main__":
    main()