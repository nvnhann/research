#include <wx/filename.h>
#include <wx/colour.h>
#include <wx/image.h>
#include <string>
#include "chatbot.h"
#include "chatlogic.h"
#include "chatgui.h"

// Dimensions of chatbot window
const int width = 414;
const int height = 736;

// wxWidgets application implementation
IMPLEMENT_APP(ChatBotApp);

std::string dataPath = "../";
std::string imgBasePath = dataPath + "images/";

bool ChatBotApp::OnInit()
{
    // Create and show the main window
    ChatBotFrame *chatBotFrame = new ChatBotFrame(wxT("Udacity ChatBot"));
    chatBotFrame->Show(true);

    return true;
}

// Main frame for ChatBot application
ChatBotFrame::ChatBotFrame(const wxString &title) 
    : wxFrame(NULL, wxID_ANY, title, wxDefaultPosition, wxSize(width, height))
{
    // Create main panel with background image
    ChatBotFrameImagePanel *ctrlPanel = new ChatBotFrameImagePanel(this);

    // Create dialog panel for chatbot conversation
    _panelDialog = new ChatBotPanelDialog(ctrlPanel, wxID_ANY);

    // Create text control for user input
    int idTextCtrl = 1;
    _userTextCtrl = new wxTextCtrl(ctrlPanel, idTextCtrl, "", wxDefaultPosition, wxSize(width, 50), wxTE_PROCESS_ENTER);
    Connect(idTextCtrl, wxEVT_TEXT_ENTER, wxCommandEventHandler(ChatBotFrame::OnEnter));

    // Arrange panels using a vertical sizer
    wxBoxSizer *vertBoxSizer = new wxBoxSizer(wxVERTICAL);
    vertBoxSizer->AddSpacer(90);
    vertBoxSizer->Add(_panelDialog, 6, wxEXPAND | wxALL, 0);
    vertBoxSizer->Add(_userTextCtrl, 1, wxEXPAND | wxALL, 5);
    ctrlPanel->SetSizer(vertBoxSizer);

    // Center window on the screen
    this->Centre();
}

void ChatBotFrame::OnEnter(wxCommandEvent &WXUNUSED(event))
{
    // Retrieve and process user input
    wxString userText = _userTextCtrl->GetLineText(0);

    // Add user text to dialog panel
    _panelDialog->AddDialogItem(userText, true);

    // Clear user input
    _userTextCtrl->Clear();

    // Send user text to chatbot
    _panelDialog->GetChatLogicHandle()->SendMessageToChatbot(std::string(userText.mb_str()));
}

BEGIN_EVENT_TABLE(ChatBotFrameImagePanel, wxPanel)
EVT_PAINT(ChatBotFrameImagePanel::paintEvent)
END_EVENT_TABLE()

ChatBotFrameImagePanel::ChatBotFrameImagePanel(wxFrame *parent) 
    : wxPanel(parent)
{
}

void ChatBotFrameImagePanel::paintEvent(wxPaintEvent &evt)
{
    wxPaintDC dc(this);
    render(dc);
}

void ChatBotFrameImagePanel::paintNow()
{
    wxClientDC dc(this);
    render(dc);
}

void ChatBotFrameImagePanel::render(wxDC &dc)
{
    // Load background image
    wxString imgFile = imgBasePath + "sf_bridge.jpg";
    wxImage image;
    image.LoadFile(imgFile);

    // Rescale image to fit window dimensions
    wxSize sz = this->GetSize();
    wxImage imgSmall = image.Rescale(sz.GetWidth(), sz.GetHeight(), wxIMAGE_QUALITY_HIGH);
    _image = wxBitmap(imgSmall);
    
    dc.DrawBitmap(_image, 0, 0, false);
}

BEGIN_EVENT_TABLE(ChatBotPanelDialog, wxPanel)
EVT_PAINT(ChatBotPanelDialog::paintEvent)
END_EVENT_TABLE()

ChatBotPanelDialog::ChatBotPanelDialog(wxWindow *parent, wxWindowID id)
    : wxScrolledWindow(parent, id)
{
    std::cout << "ChatBotPanelDialog Constructor" << std::endl;
    // Initialize sizer for dialog items
    _dialogSizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(_dialogSizer);

    // Allow PNG images to be used
    wxInitAllImageHandlers();

    // Create and initialize ChatLogic instance
    _chatLogic = std::make_unique<ChatLogic>();

    // Set dialog handle in ChatLogic
    _chatLogic->SetPanelDialogHandle(this);

    // Load answer graph from file
    _chatLogic->LoadAnswerGraphFromFile(dataPath + "src/answergraph.txt");
}

ChatBotPanelDialog::~ChatBotPanelDialog()
{
    std::cout << "ChatBotPanelDialog Destructor" << std::endl;
}

void ChatBotPanelDialog::AddDialogItem(wxString text, bool isFromUser)
{
    // Add a new dialog item to the sizer
    ChatBotPanelDialogItem *item = new ChatBotPanelDialogItem(this, text, isFromUser);
    _dialogSizer->Add(item, 0, wxALL | (isFromUser ? wxALIGN_LEFT : wxALIGN_RIGHT), 8);
    _dialogSizer->Layout();

    // Adjust scroll settings
    this->FitInside(); 
    this->SetScrollRate(5, 5);
    this->Layout();

    // Scroll to the bottom to show the newest item
    int dx, dy;
    this->GetScrollPixelsPerUnit(&dx, &dy);
    int sy = dy * this->GetScrollLines(wxVERTICAL);
    this->DoScroll(0, sy);
}

void ChatBotPanelDialog::PrintChatbotResponse(std::string response)
{
    // Convert response to wxString and add dialog item
    wxString botText(response.c_str(), wxConvUTF8);
    AddDialogItem(botText, false);
}

void ChatBotPanelDialog::paintEvent(wxPaintEvent &evt)
{
    wxPaintDC dc(this);
    render(dc);
}

void ChatBotPanelDialog::paintNow()
{
    wxClientDC dc(this);
    render(dc);
}

void ChatBotPanelDialog::render(wxDC &dc)
{
    wxImage image;
    image.LoadFile(imgBasePath + "sf_bridge_inner.jpg");

    wxSize sz = this->GetSize();
    wxImage imgSmall = image.Rescale(sz.GetWidth(), sz.GetHeight(), wxIMAGE_QUALITY_HIGH);

    _image = wxBitmap(imgSmall);
    dc.DrawBitmap(_image, 0, 0, false);
}

ChatBotPanelDialogItem::ChatBotPanelDialogItem(wxPanel *parent, wxString text, bool isFromUser)
    : wxPanel(parent, -1, wxPoint(-1, -1), wxSize(-1, -1), wxBORDER_NONE)
{
    std::cout << "Creating new ChatBotPanelDialogItem" << std::endl;
    // Retrieve image from chatbot if not user
    wxBitmap *bitmap = isFromUser ? nullptr : ((ChatBotPanelDialog*)parent)->GetChatLogicHandle()->GetImageFromChatbot();

    // Create image and text elements
    _chatBotImg = new wxStaticBitmap(this, wxID_ANY, (isFromUser ? wxBitmap(imgBasePath + "user.png", wxBITMAP_TYPE_PNG) : *bitmap), wxPoint(-1, -1), wxSize(-1, -1));
    _chatBotTxt = new wxStaticText(this, wxID_ANY, text, wxPoint(-1, -1), wxSize(150, -1), wxALIGN_CENTRE | wxBORDER_NONE);
    _chatBotTxt->SetForegroundColour(isFromUser ? wxColor(*wxBLACK) : wxColor(*wxWHITE));

    // Arrange elements with a horizontal sizer
    wxBoxSizer *horzBoxSizer = new wxBoxSizer(wxHORIZONTAL);
    horzBoxSizer->Add(_chatBotTxt, 8, wxEXPAND | wxALL, 1);
    horzBoxSizer->Add(_chatBotImg, 2, wxEXPAND | wxALL, 1);
    this->SetSizer(horzBoxSizer);

    // Wrap text after 150 pixels
    _chatBotTxt->Wrap(150);

    // Set background color
    this->SetBackgroundColour((isFromUser ? wxT("YELLOW") : wxT("BLUE")));
}