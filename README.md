# SmartQuery: Search Engine using Tools and Agents

This project is a chatbot that leverages LangChain agents to search and retrieve information from multiple sources, including Wikipedia, Arxiv, DuckDuckGo, and a custom Formula 1 Sporting Regulations PDF.

## Features:
- **Wikipedia Search**: Retrieve information from Wikipedia.
- **Arxiv Search**: Access academic papers and articles from Arxiv.
- **DuckDuckGo Search**: Perform web searches.
- **Custom PDF Search**: Search within a Formula 1 Sporting Regulations document.

### How it Works:
1. **User Input**: The user can ask questions via the chat interface.
2. **Tools**: The app uses various LangChain tools to search through the available sources.
3. **Response**: The chatbot replies with information retrieved from the sources.

### Setup Instructions:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/langchain-chat-search.git
   cd langchain-chat-search
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your `.env` file with your **Groq API Key**:
   - Create a `.env` file in the root directory.
   - Add your **API key** like so:
     ```
     OPENAI_API_KEY=your-api-key
     ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Available Tools:
- **Wikipedia API**: Used for querying Wikipedia.
- **Arxiv API**: Used to query academic papers from Arxiv.
- **DuckDuckGo Search**: For general web search queries.
- **Custom Formula 1 PDF**: Search within the Formula 1 Sporting Regulations PDF.

### Example Questions:
- **General**: "What is the Drag Reduction System (DRS) in Formula 1?"
- **Qualifying**: "How is the starting grid determined if a qualifying session is interrupted?"
- **Penalties**: "What happens if a driver fails to set a time in any of the qualifying sessions?"
- **Safety**: "What procedures are followed when a red flag is shown during a race?"

---

### Dependencies:
- **Streamlit**: For building the user interface.
- **LangChain**: For integrating the various tools and AI models.
- **DuckDuckGo**: For searching the web.
- **Arxiv API**: For retrieving academic papers.
- **Wikipedia API**: For querying Wikipedia articles.
- **Groq API**: For handling AI-related requests.

---
### License:
This project is licensed under the MIT License.
