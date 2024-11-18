#ifndef GRAPHNODE_H_
#define GRAPHNODE_H_

#include <vector>
#include <string>
#include <memory>
#include "chatbot.h"

// Forward declarations
class GraphEdge;

class GraphNode
{
private:
    // Data handles (owned)
    std::vector<std::unique_ptr<GraphEdge>> _childEdges; // Edges to subsequent nodes

    // Data handles (not owned)
    std::vector<GraphEdge *> _parentEdges; // Edges to preceding nodes
    ChatBot _chatBot;

    // Proprietary members
    int _id;
    std::vector<std::string> _answers;

public:
    // Constructor / destructor
    GraphNode(int id);
    ~GraphNode();

    // Getter / setter
    int GetID() const { return _id; }
    int GetNumberOfChildEdges() const { return _childEdges.size(); }
    GraphEdge *GetChildEdgeAtIndex(int index);
    std::vector<std::string> GetAnswers() const { return _answers; }
    int GetNumberOfParents() const { return _parentEdges.size(); }

    // Proprietary functions
    void AddToken(const std::string &token); // Add answers to list
    void AddEdgeToParentNode(GraphEdge *edge);
    void AddEdgeToChildNode(std::unique_ptr<GraphEdge> edge);

    void MoveChatbotHere(ChatBot chatbot);
    void MoveChatbotToNewNode(GraphNode *newNode);
};

#endif /* GRAPHNODE_H_ */