# GestureBrowseAI

GestureBrowseAI is an intelligent agentic AI application that combines natural language understanding, autonomous web navigation, and real-time gesture-based interaction. Users simply enter a query, and the system searches the web through DuckDuckGo, opens the relevant page using Selenium, and enables hands-free browsing through gesture-based scrolling powered by computer vision.

## Features

- Accepts user queries in natural language using LangChain’s agentic architecture  
- Performs real-time web search using DuckDuckGo for privacy-respecting results  
- Automatically opens the top-ranked web page through Selenium-based browser control  
- Uses webcam input to detect hand gestures for scroll commands (up/down)  
- Smooth, hands-free web interaction experience combining AI, automation, and vision  

## Technologies Used

- **LangChain** – for intelligent agent control and query processing  
- **DuckDuckGo Search** – for web search operations  
- **Selenium WebDriver** – for automated browser control and interaction  
- **OpenCV** – for real-time computer vision processing  
- **Mediapipe** – for high-accuracy hand tracking and gesture recognition  
- **Python** – core language powering all logic and integrations  

## Workflow

1. The user inputs a search query through a command interface  
2. LangChain processes the query and determines the search strategy  
3. DuckDuckGo returns search results, and the top link is opened via Selenium  
4. The system activates the webcam and starts detecting hand gestures  
5. Based on the gesture detected, the web page scrolls up or down automatically  

## Applications

- Hands-free web browsing for presentations, demos, and accessibility tools  
- AI-powered research assistants and productivity enhancers  
- Vision-based browser control for custom UI/UX experiments  
- Integrations for assistive technologies using gesture recognition  
