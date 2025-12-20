#!/usr/bin/env python3
"""
Interactive SQL query tool for LanceDB vector database using DuckDB.
Allows running SQL queries over the embedded documents.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import lancedb
from tools import retriever


def display_schema(conn):
    """Display the schema of available tables."""
    print("\n" + "="*60)
    print("Available Tables and Schema")
    print("="*60 + "\n")
    
    tables = conn.execute("SHOW TABLES").fetchall()
    for table in tables:
        table_name = table[0]
        print(f"Table: {table_name}")
        print("-" * 40)
        schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
        for col in schema:
            print(f"  {col[0]:<20} {col[1]}")
        print()


def run_query(conn, query):
    """Run a SQL query and display results."""
    try:
        result = conn.execute(query).fetchall()
        columns = [desc[0] for desc in conn.description]
        
        # Display results
        print("\n" + "="*80)
        print("Query Results")
        print("="*80 + "\n")
        
        if not result:
            print("No results found.")
            return
        
        # Calculate max width for each column
        col_widths = []
        for i, col in enumerate(columns):
            max_width = len(col)
            for row in result:
                max_width = max(max_width, len(str(row[i])))
            col_widths.append(max_width)
        
        # Print header
        header = " | ".join(f"{col:<{col_widths[i]}}" for i, col in enumerate(columns))
        print(header)
        print("-" * len(header))
        
        # Print rows
        for row in result:
            row_str = " | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row))
            print(row_str)
        
        print(f"\nTotal rows: {len(result)}")
        
    except Exception as e:
        print(f"\nError executing query: {e}")


def interactive_mode(conn):
    """Run in interactive query mode."""
    print("\n" + "="*60)
    print("Interactive SQL Query Mode")
    print("="*60)
    print("\nCommands:")
    print("  schema        - Show table schema")
    print("  exit/quit     - Exit the program")
    print("  Any SQL query - Execute the query")
    print("\nExamples:")
    print("  SELECT COUNT(*) FROM embeddings;")
    print("  SELECT source, sequence_num, LENGTH(text) as len FROM embeddings LIMIT 5;")
    print("  SELECT DISTINCT source FROM embeddings;")
    print("  SELECT * FROM embeddings WHERE source LIKE '%wikipedia%';")
    print()
    
    while True:
        try:
            query = input("\nSQL> ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
            
            if query.lower() == 'schema':
                display_schema(conn)
                continue
            
            run_query(conn, query)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


def main():
    """Main function to set up DuckDB connection to LanceDB."""
    
    # Connect to LanceDB
    db_path = Path(__file__).parent.parent / "data" / "lancedb"
    
    if not db_path.exists():
        print(f"Error: LanceDB database not found at {db_path}")
        print("Run test_retriever.py first to create and populate the database.")
        sys.exit(1)
    
    print(f"\nConnecting to LanceDB at: {db_path}")
    
    # Connect to LanceDB
    lance_db = lancedb.connect(str(db_path))
    
    # Check if embeddings table exists
    try:
        table = lance_db.open_table(retriever.TABLE_NAME)
    except Exception as e:
        print(f"\nError: Could not open table '{retriever.TABLE_NAME}': {e}")
        print("Run test_retriever.py first to create and populate the database.")
        sys.exit(1)
    
    # Convert to Arrow table for DuckDB
    arrow_table = table.to_arrow()
    
    # Create DuckDB connection
    conn = duckdb.connect(':memory:')
    
    # Register the LanceDB table as a DuckDB table
    conn.register('embeddings', arrow_table)
    
    print(f"Successfully loaded {len(arrow_table)} rows from LanceDB")
    
    # Get basic stats
    stats = conn.execute("""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT source) as unique_sources,
            AVG(LENGTH(text)) as avg_text_length,
            MAX(sequence_num) + 1 as max_chunks
        FROM embeddings
    """).fetchone()
    
    print(f"\nDatabase Statistics:")
    print(f"  Total rows: {stats[0]}")
    print(f"  Unique sources: {stats[1]}")
    print(f"  Average text length: {stats[2]:.0f} characters")
    print(f"  Max chunks per document: {stats[3]}")
    
    # Show sample data
    print("\n" + "="*60)
    print("Sample Data (First 3 rows)")
    print("="*60 + "\n")
    
    sample = conn.execute("""
        SELECT id, source, sequence_num, LENGTH(text) as text_length
        FROM embeddings
        LIMIT 3
    """).fetchall()
    
    for row in sample:
        print(f"ID: {row[0]}")
        print(f"Source: {row[1]}")
        print(f"Sequence: {row[2]}")
        print(f"Text length: {row[3]} chars")
        print("-" * 40)
    
    # Enter interactive mode
    interactive_mode(conn)


if __name__ == "__main__":
    main()
