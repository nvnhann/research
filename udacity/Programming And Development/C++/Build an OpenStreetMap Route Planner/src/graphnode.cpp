#include "graphedge.h"
#include "graphnode.h"

GraphNode::GraphNode(int id) : _id(id)
{
}

GraphNode::~GraphNode()
{
    //// STUDENT CODE
    ////
    // Không cần delete _chatBot vì `_chatBot` giờ là một đối tượng, không còn là con trỏ
    ////
    //// EOF STUDENT CODE
}

void GraphNode::AddToken(const std::string &token)
{
    _answers.push_back(token);
}

void GraphNode::AddEdgeToParentNode(GraphEdge *edge)
{
    _parentEdges.push_back(edge);
}

void GraphNode::AddEdgeToChildNode(std::unique_ptr<GraphEdge> edge)
{
    _childEdges.push_back(std::move(edge));
}

//// STUDENT CODE
////
void GraphNode::MoveChatbotHere(ChatBot chatbot)
{
    _chatBot = std::move(chatbot);
    _chatBot.SetCurrentNode(this);
}

void GraphNode::MoveChatbotToNewNode(GraphNode *newNode)
{
    newNode->MoveChatbotHere(std::move(_chatBot));
    //_chatBot = nullptr; // không cần thiết vì `_chatBot` giờ là một đối tượng
}
////
//// EOF STUDENT CODE

GraphEdge *GraphNode::GetChildEdgeAtIndex(int index)
{
    return _childEdges[index].get();
}