#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <memory>

#include "graphedge.h"
#include "graphnode.h"
#include "chatbot.h"
#include "chatlogic.h"

ChatLogic::ChatLogic()
{
    std::cout << "ChatLogic Constructor" << std::endl;
}

ChatLogic::~ChatLogic()
{
    std::cout << "ChatLogic Destructor" << std::endl;
}

template <typename T>
void ChatLogic::AddAllTokensToElement(std::string tokenID, tokenlist &tokens, T &element)
{
    auto token = tokens.begin();
    while (true)
    {
        token = std::find_if(token, tokens.end(), [&tokenID](const std::pair<std::string, std::string> &pair) { return pair.first == tokenID; });
        if (token != tokens.end())
        {
            element.AddToken(token->second);
            token++;
        }
        else
        {
            break;
        }
    }
}

void ChatLogic::LoadAnswerGraphFromFile(std::string filename)
{
    std::ifstream file(filename);
    if (!file)
    {
        std::cout << "File could not be opened!" << std::endl;
        return;
    }

    std::string lineStr;
    while (std::getline(file, lineStr))
    {
        tokenlist tokens;
        while (!lineStr.empty())
        {
            int posTokenFront = lineStr.find("<");
            int posTokenBack = lineStr.find(">");
            if (posTokenFront < 0 || posTokenBack < 0)
                break;
            std::string tokenStr = lineStr.substr(posTokenFront + 1, posTokenBack - posTokenFront - 1);

            int posTokenInfo = tokenStr.find(":");
            if (posTokenInfo != std::string::npos)
            {
                std::string tokenType = tokenStr.substr(0, posTokenInfo);
                std::string tokenInfo = tokenStr.substr(posTokenInfo + 1);

                tokens.push_back(std::make_pair(tokenType, tokenInfo));
            }

            lineStr = lineStr.substr(posTokenBack + 1);
        }

        auto type = std::find_if(tokens.begin(), tokens.end(), [](const std::pair<std::string, std::string> &pair) {
            return pair.first == "TYPE";
        });

        if (type != tokens.end())
        {
            auto idToken = std::find_if(tokens.begin(), tokens.end(), [](const std::pair<std::string, std::string> &pair) {
                return pair.first == "ID";
            });

            if (idToken != tokens.end())
            {
                int id = std::stoi(idToken->second);

                if (type->second == "NODE")
                {
                    auto newNode = std::find_if(_nodes.begin(), _nodes.end(), [&id](std::unique_ptr<GraphNode> &node) {
                        return node->GetID() == id;
                    });

                    if (newNode == _nodes.end())
                    {
                        _nodes.emplace_back(std::make_unique<GraphNode>(id));
                        newNode = _nodes.end() - 1;

                        AddAllTokensToElement("ANSWER", tokens, **newNode);
                    }
                }

                if (type->second == "EDGE")
                {
                    auto parentToken = std::find_if(tokens.begin(), tokens.end(), [](const std::pair<std::string, std::string> &pair) {
                        return pair.first == "PARENT";
                    });
                    auto childToken = std::find_if(tokens.begin(), tokens.end(), [](const std::pair<std::string, std::string> &pair) {
                        return pair.first == "CHILD";
                    });

                    if (parentToken != tokens.end() && childToken != tokens.end())
                    {
                        auto parentNode = std::find_if(_nodes.begin(), _nodes.end(), [&parentToken](std::unique_ptr<GraphNode> &node) {
                            return node->GetID() == std::stoi(parentToken->second);
                        });
                        auto childNode = std::find_if(_nodes.begin(), _nodes.end(), [&childToken](std::unique_ptr<GraphNode> &node) {
                            return node->GetID() == std::stoi(childToken->second);
                        });

                        std::unique_ptr<GraphEdge> edge = std::make_unique<GraphEdge>(id);
                        edge->SetChildNode((*childNode).get());
                        edge->SetParentNode((*parentNode).get());

                        AddAllTokensToElement("KEYWORD", tokens, *edge);

                        (*childNode)->AddEdgeToParentNode(edge.get());
                        (*parentNode)->AddEdgeToChildNode(std::move(edge));
                    }
                }
            }
            else
            {
                std::cout << "Error: ID missing. Line is ignored!" << std::endl;
            }
        }
    }

    file.close();

    GraphNode *rootNode = nullptr;
    for (auto &node : _nodes)
    {
        if (node->GetNumberOfParents() == 0)
        {
            if (rootNode == nullptr)
            {
                rootNode = node.get();
            }
            else
            {
                std::cout << "ERROR : Multiple root nodes detected" << std::endl;
            }
        }
    }

    ChatBot chatBot("../images/chatbot.png");
    SetChatbotHandle(&chatBot);
    chatBot.SetChatLogicHandle(this);
    chatBot.SetRootNode(rootNode);
    rootNode->MoveChatbotHere(std::move(chatBot));
}

void ChatLogic::SetPanelDialogHandle(ChatBotPanelDialog *panelDialog)
{
    _panelDialog = panelDialog;
}

void ChatLogic::SetChatbotHandle(ChatBot *chatbot)
{
    _chatBot = chatbot;
}

void ChatLogic::SendMessageToChatbot(std::string message)
{
    _chatBot->ReceiveMessageFromUser(std::move(message));
}

void ChatLogic::SendMessageToUser(std::string message)
{
    _panelDialog->PrintChatbotResponse(std::move(message));
}

wxBitmap *ChatLogic::GetImageFromChatbot()
{
    return _chatBot->GetImageHandle();
}