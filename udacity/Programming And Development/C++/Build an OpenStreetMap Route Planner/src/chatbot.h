#ifndef CHATBOT_H_
#define CHATBOT_H_

#include <wx/bitmap.h>
#include <string>

class GraphNode; // forward declaration
class ChatLogic; // forward declaration

class ChatBot
{
private:
    // Avatar image stored in heap
    wxBitmap *_image; 

    // Pointers to other objects (not owned)
    GraphNode *_currentNode;
    GraphNode *_rootNode;
    ChatLogic *_chatLogic;

    // function to compute Levenshtein distance between two strings
    int ComputeLevenshteinDistance(std::string s1, std::string s2);

public:
    // Constructors and destructor
    ChatBot();                     // Default constructor without memory allocation
    ChatBot(std::string filename); // Constructor with memory allocation
    ~ChatBot();

    // Rule of Five
    // Copy constructor
    ChatBot(const ChatBot &source); 
    // Copy assignment operator
    ChatBot &operator=(const ChatBot &source); 
    // Move constructor 
    ChatBot(ChatBot &&source); 
    // Move assignment operator
    ChatBot &operator=(ChatBot &&source); 

    // Setters
    void SetCurrentNode(GraphNode *node);
    void SetRootNode(GraphNode *rootNode) { _rootNode = rootNode; }
    void SetChatLogicHandle(ChatLogic *chatLogic) { _chatLogic = chatLogic; }
    // Getter
    wxBitmap *GetImageHandle() { return _image; }

    // Function to process user messages
    void ReceiveMessageFromUser(std::string message);
};

#endif /* CHATBOT_H_ */