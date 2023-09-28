from flask import Flask, render_template, jsonify, request
from whoosh.qparser import MultifieldParser, OrGroup, PhrasePlugin
from whoosh.fields import Schema, TEXT, ID
from whoosh import index
from openai.error import OpenAIError
import openai
import logging
import logging.handlers
from collections import defaultdict
import os
import re
import werkzeug.exceptions
from logtail import LogtailHandler






BASE_DIR = os.path.abspath(os.path.dirname(__file__))
index_dir = os.path.join(BASE_DIR, "indexdir")
bylaws_dir = os.path.join(BASE_DIR, "table_extract")



openai.api_key = 'OPENAI_API_KEY'

app = Flask(__name__)

handler = LogtailHandler(source_token="fLYjHRS3uJzPh1Xk5eErS5kA")  # replace with your actual source token
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Logging setup
handler = logging.handlers.SysLogHandler(address=("logs-01.loggly.com", 514))
formatter = logging.Formatter(
    'Python: { "loggerName":"%(name)s", "timestamp":"%(asctime)s", "pathName":"%(pathname)s", "logRecordCreationTime":"%(created)f", "functionName":"%(funcName)s", "levelNo":"%(levelno)s", "lineNo":"%(lineno)d", "time":"%(msecs)d", "levelName":"%(levelname)s", "message":"%(message)s"}'
)
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Initialize Whoosh index
schema = Schema(title=ID(stored=True), content=TEXT(stored=True))
index_dir = "indexdir"

if not os.path.exists(index_dir):
    os.mkdir(index_dir)
    ix = index.create_in(index_dir, schema)
else:
    ix = index.open_dir(index_dir)




# This function uses GPT-3 to get essential keywords from a question
# This function uses GPT-3 to get essential keywords from a question
def get_keywords(question):
    try:
        messages = [
            {"role": "system", "content": "You are a text-based assistant trained to provide information based on specific text content."},
            {"role": "assistant", "content": "Please identify the essential keywords in the following question."},
            {"role": "user", "content": question}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=0.2,
            max_tokens=1000
        )
        
        extracted_keywords = response.choices[0].message['content']


        # Check if extracted_keywords is a string
        if isinstance(extracted_keywords, str):
            keywords = [keyword.strip() for keyword in extracted_keywords.split(",")]
        else:
            keywords = extracted_keywords  # assuming it's already a list

        if not keywords:
            app.logger.warning("No keywords extracted.")
            return []
        
        return keywords


    except OpenAIError as e:
        app.logger.error(f"OpenAI API call failed during keyword extraction: {e}")
        raise

def needs_indexing():
    with ix.searcher() as searcher:
        docnum = len(list(searcher.documents()))
        return docnum == 0

def index_text_files():
    writer = ix.writer()
    bylaws_dir = os.path.join(BASE_DIR, "table_extract")
    for filename in os.listdir(bylaws_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(bylaws_dir, filename)
            with open(filepath, encoding="utf8") as f:
                text = f.read()
            writer.add_document(title=filename, content=text)
    writer.commit()

def initialize_whoosh():
    global ix
    schema = Schema(title=ID(stored=True), content=TEXT(stored=True))
    index_dir = "indexdir"

    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
        ix = index.create_in(index_dir, schema)
    else:
        ix = index.open_dir(index_dir)

    # Index text files during startup only if needed
    if needs_indexing():
        index_text_files()

# Call the initialization function
initialize_whoosh()


# Initialize Whoosh index
schema = Schema(title=ID(stored=True), content=TEXT(stored=True))
index_dir = "indexdir"

if not os.path.exists(index_dir):
    os.mkdir(index_dir)
    ix = index.create_in(index_dir, schema)
else:
    ix = index.open_dir(index_dir)

# Index text files during startup only if needed
if needs_indexing():
    index_text_files()


    

# This function uses GPT-3 to provide a final answer based on the relevant content
def get_answer(question, relevant_content):
    messages = [
        {
            "role": "system",
            "content": "You are a text-based assistant trained to provide information based on specific text content."
        },
        {
            "role": "assistant",
            "content": f"The relevant content is: {relevant_content[:10000]}..."  # truncate for brevity
        },
        {
            "role": "user",
            "content": question
        }
    ]
    try:
        app.logger.debug(f"Messages sent to OpenAI for generating an answer: {messages}")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=1,
            max_tokens=1000
        )
        answer = response.choices[0].message['content']

        if not answer:
            app.logger.warning("No answer generated.")
            return ""

    except OpenAIError as e:
        app.logger.error(f'OpenAI API call failed: {e}')
        return None

    return answer


# Post-Processing: Extract paragraphs containing all keywords
def post_process_content(content, keywords):
    if not isinstance(content, str):
        app.logger.error(f"Content is not a string: {content}")
        return ""
    paragraphs = content.split('\n')
    filtered_paragraphs = [p for p in paragraphs if all(k.lower() in p.lower() for k in keywords)]
    return '\n'.join(filtered_paragraphs)


# Modified function to use more refined Whoosh query types and increased token limit
def get_surrounding_content(content, best_positions, before=1000, after=1000):
    if isinstance(best_positions, int) or best_positions is None:
        app.logger.warning(f"Invalid best_positions provided: {best_positions}")
        return ""

    app.logger.info(f"Getting surrounding content for positions: {best_positions}")
    start_position = max(0, best_positions[0] - before)
    end_position = min(len(content), best_positions[1] + after)

    surrounding_content = content[start_position:end_position]
    if not surrounding_content.strip():
        app.logger.warning("Extracted surrounding content is empty.")
        return ""

    app.logger.info(f"Extracted surrounding content: {surrounding_content[:800]}...")  
    return surrounding_content


def get_best_positions(terms_positions):

    """
    Given a dictionary with terms as keys and their positions as values, 
    find the smallest window that contains all terms.
    
    :param terms_positions: Dict[str, List[int]] 
        A dictionary where key is a term and value is the list of positions for that term.
    :return: Tuple[int, int]
        A tuple representing the start and end of the best position.
    """

    # Convert the dictionary into a list of (position, term) pairs and sort it
    sorted_positions = sorted(
        [(position, term) for term, positions in terms_positions.items() for position in positions]
    )

    # Initialize the pointers and variables for the sliding window
    left, right = 0, 0
    current_terms = {}
    min_window_size = float('inf')
    best_window = None

    while right < len(sorted_positions):
        # Expand the window to the right
        term = sorted_positions[right][1]
        current_terms[term] = current_terms.get(term, 0) + 1
        right += 1

        # Check if the window contains all terms
        while len(current_terms) == len(terms_positions):
            # Update the best window if the current window is smaller
            window_size = sorted_positions[right - 1][0] - sorted_positions[left][0]
            if window_size < min_window_size:
                min_window_size = window_size
                best_window = (sorted_positions[left][0], sorted_positions[right - 1][0])

            # Try to reduce the window from the left
            term = sorted_positions[left][1]
            current_terms[term] -= 1
            if current_terms[term] == 0:
                del current_terms[term]
            left += 1

    if best_window:
        return best_window
    else:
        return None


from itertools import combinations





def search_content(query, bylaw_file, max_results=10):
    if isinstance(query, list):
        app.logger.info(f"Received question: {', '.join(query)}")
    else:
        app.logger.info(f"Received question: {query}")

    app.logger.info(f"Received bylaw_file: {bylaw_file}")

    filepath = find_bylaw_file(bylaw_file)
    app.logger.info(f"Reading file: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        debug_content = f.read(10000)
    app.logger.info(f"Debugging content: {debug_content[:800]}")#_______- CHange it from 500) Changed to 800

    fields = ["title", "content"]
    group = OrGroup.factory(0.9)
    parser = MultifieldParser(fields, ix.schema, group=group)
    parser.add_plugin(PhrasePlugin())

    terms = [f'"{term}"~101' for term in query]
    combined_query = " ".join(terms)
    my_query = parser.parse(combined_query)

    app.logger.info(f"Whoosh Query: {my_query}")

    combined_content = ""
    with ix.searcher() as searcher:
        results = searcher.search(my_query, limit=max_results)
        app.logger.info(f"Whoosh results: {results}")

        if not results:
            app.logger.warning(f"No content found for query: {query}")
            return ""

        for result in results:
            if isinstance(query, list):
                terms_to_check = query
            else:
                terms_to_check = query.split()

            combined_text = result["title"] + " " + result["content"]

            # Calculate best positions using find_closest_proximity
            # Calculate best positions using best_positions
            terms_positions = {keyword: [m.start() for m in re.finditer(re.escape(keyword), combined_text, re.IGNORECASE)] for keyword in terms_to_check}
            best_positions_result = get_best_positions(terms_positions)

        

            if best_positions_result:
                surrounding_content = get_surrounding_content(combined_text, best_positions_result)

                combined_content += surrounding_content + " "
            
            if isinstance(query, list):
                terms_to_log = ', '.join(query)
            else:
                terms_to_log = query

            app.logger.info(f"Surrounding content for terms {terms_to_log}: {surrounding_content[:500]}...")

    app.logger.info(f"Combined relevant content: {combined_content[:1000]}")
    return combined_content.strip()





def rename_files():
    bylaws_dir = os.path.join(BASE_DIR, "table_extract")

    for filename in os.listdir(bylaws_dir):
        if filename[0] == ' ':
            source = os.path.join(bylaws_dir, filename)
            destination = os.path.join(bylaws_dir, filename[1:])
            os.rename(source, destination)

rename_files()

def get_bylaw_names():
    basedir = os.path.abspath(os.path.dirname(__file__))
    bylaws_dir = os.path.join(basedir, 'table_extract')
    bylaw_files = [f[:-4].strip() for f in sorted(os.listdir(bylaws_dir)) if f.endswith('.txt')]

    bylaws = defaultdict(list)
    for bylaw in bylaw_files:
        bylaws[bylaw[0].upper()].append(bylaw.replace('_', ' '))

    for k in bylaws:
        bylaws[k] = sorted(bylaws[k])

    return dict(bylaws)

def find_bylaw_file(bylaw_name):
    basedir = os.path.abspath(os.path.dirname(__file__))
    bylaws_dir = os.path.join(basedir, 'table_extract')
    bylaw_files = [f for f in os.listdir(bylaws_dir) if f.endswith('.txt')]

    for file_path in bylaw_files:
        if bylaw_name.lower() in file_path.replace('_', ' ').lower():
            return os.path.join(bylaws_dir, file_path)

    raise FileNotFoundError(f"The content file {bylaw_name} does not exist.")

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/api/bylaws', methods=['GET'])
def bylaws():
    bylaws = get_bylaw_names()
    return jsonify(bylaws)

@app.route('/api', methods=['POST'])
def api():
    try:
        data = request.get_json()
        question = data.get('question')
        bylaw_file = data.get('bylaw')

        if not bylaw_file:
            return jsonify({'error': "Please select a file on the left."}), 400

        # Obtain the filepath corresponding to the bylaw_file
        filepath = find_bylaw_file(bylaw_file)

        # Step 1: Extract essential keywords using GPT-3
        keywords = get_keywords(question)
        if not keywords:
            app.logger.error("Failed to extract keywords from the provided question.")
            raise ValueError('Failed to extract keywords')

        # Step 2: Use keywords to find relevant content
        relevant_content = search_content(keywords, bylaw_file)
        if not relevant_content:
            app.logger.error("Failed to find relevant content for the extracted keywords.")
            raise ValueError('No relevant content found')

        # Extract surrounding content for the positions
        terms_positions = {keyword: [m.start() for m in re.finditer(re.escape(keyword), relevant_content, re.IGNORECASE)] for keyword in keywords}
        best_positions_result = get_best_positions(terms_positions)
        if not best_positions_result:
            app.logger.error("Failed to find best positions for the keywords.")
            raise ValueError('Failed to find best positions')

        extracted_content = get_surrounding_content(relevant_content, best_positions_result)
        print(f"Content around best positions {best_positions_result}:\n")
        print(extracted_content)
        print("-" * 80)

        # Step 3: Use GPT-3 to generate a contextual answer based on relevant_content and the original question
        answer = get_answer(question, extracted_content)
        if not answer:
            app.logger.error("Failed to generate an answer using the provided content and OpenAI.")
            raise ValueError('Failed to generate an answer')

        # Step 4: Return the generated answer
        return jsonify({'answer': answer})

    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500




def extract_content_around_position(filepath, position, range_=500):
    with open(filepath, 'r', encoding='utf-8') as file:
        file.seek(max(0, position - range_))  # Go to the position minus the range
        content = file.read(2 * range_)  # Read twice the range: range before and range after
    return content

# Commented out the test block as it uses hardcoded filepath and positions
# positions = [77134, 77209, 77453]
# filepath = "/Users/connorpettepiece/Desktop/OAKBAY_FP/Oak_Bay_Financial_Plan/Oak_Bay_Financial_Plan_2020_to_2024.txt"

# for position in positions:
#     extracted_content = extract_content_around_position(filepath, position)
#     print(f"Content around position {position}:\n")
#     print(extracted_content)
#     print("-" * 80)


@app.route('/path')
def path():
    return os.getcwd()

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({'error': str(e)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error for debugging
    app.logger.error(f"Internal error: {e}")
    
    # Check if it's a 400 BAD REQUEST error
    if isinstance(e, werkzeug.exceptions.BadRequest):
        return jsonify({'error': str(e)}), 400
    
    # Return user-friendly message for all other exceptions
    return jsonify({'error': "Sorry, something went wrong! Please try again."}), 500



if __name__ == "__main__":
    app.run(debug=True)
