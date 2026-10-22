// MainFrm.cpp : implementation of the CMainFrame class
//

#include "stdafx.h"
#include "PlugInArchitecture.h"

#include "MainFrm.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CMainFrame

IMPLEMENT_DYNAMIC(CMainFrame, CMDIFrameWnd)

BEGIN_MESSAGE_MAP(CMainFrame, CMDIFrameWnd)
	ON_WM_CREATE()
	ON_COMMAND_RANGE(ID_PLUGINS_CMD, ID_PLUGINS_CMD_LIMIT, OnMenuExcute)
	/* ON_COMMAND(ID_TOOLBAR_CMD,   OnToolBT1)
	 ON_COMMAND(ID_TOOLBAR_CMD+1, OnToolBT2)
	 ON_COMMAND(ID_TOOLBAR_CMD+2, OnToolBT3)
	 ON_COMMAND(ID_TOOLBAR_CMD+3, OnToolBT4)
	 ON_COMMAND(ID_TOOLBAR_CMD+4, OnToolBT5)*/
	 ON_COMMAND_RANGE(ID_TOOLBAR_CMD, ID_TOOLBAR_CMD_LIMIT, OnToolBarExcute)
END_MESSAGE_MAP()

static UINT indicators[] =
{
	ID_SEPARATOR,           // status line indicator
	ID_INDICATOR_CAPS,
	ID_INDICATOR_NUM,
	ID_INDICATOR_SCRL,
};


// CMainFrame construction/destruction

CMainFrame::CMainFrame()
{
	// TODO: add member initialization code here
	m_pluginManager = new PluginManager();
}

CMainFrame::~CMainFrame()
{
	delete m_pluginManager;
}


int CMainFrame::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CMDIFrameWnd::OnCreate(lpCreateStruct) == -1)
		return -1;
	
	//if (!m_wndToolBar.CreateEx(this, TBSTYLE_FLAT, WS_CHILD | WS_VISIBLE | CBRS_TOP
	//	| CBRS_GRIPPER | CBRS_TOOLTIPS | CBRS_FLYBY | CBRS_SIZE_DYNAMIC) ||
	//	!m_wndToolBar.LoadToolBar(IDR_MAINFRAME))
	//{
	//	TRACE0("Failed to create toolbar\n");
	//	return -1;      // fail to create
	//}

	if (!m_pluginToolbar.CreateEx(this, TBSTYLE_FLAT, WS_CHILD | WS_VISIBLE | CBRS_TOP
		| CBRS_GRIPPER | CBRS_TOOLTIPS | CBRS_FLYBY | CBRS_SIZE_DYNAMIC))
	{
		TRACE0("Failed to create toolbar\n");
		return -1;      // fail to create
	}

	if (!m_wndStatusBar.Create(this) ||
		!m_wndStatusBar.SetIndicators(indicators,
		  sizeof(indicators)/sizeof(UINT)))
	{
		TRACE0("Failed to create status bar\n");
		return -1;      // fail to create
	}

	// TODO: Delete these three lines if you don't want the toolbar to be dockable
	m_pluginToolbar.EnableDocking(CBRS_ALIGN_ANY);
	EnableDocking(CBRS_ALIGN_ANY);
	DockControlBar(&m_pluginToolbar);

	m_hMenu = this->GetMenu()->m_hMenu;
	PluginParameters *pNppParameters = PluginParameters::GetInstance();
	Plugin_plus plugin;
	plugin.init(m_pluginManager, m_hMenu, m_hWnd, &m_wndToolBar);
	this->m_pluginhIcon = m_pluginManager->m_pluginhIcon;
	this->m_tmphIcon = m_pluginManager->GethIcon();
	m_pluginManager->SetToolBarButtonID();
	AddButtonToToolBar();
	return 0;
}

BOOL CMainFrame::PreCreateWindow(CREATESTRUCT& cs)
{
	if( !CMDIFrameWnd::PreCreateWindow(cs) )
		return FALSE;
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return TRUE;
}


// CMainFrame diagnostics

#ifdef _DEBUG
void CMainFrame::AssertValid() const
{
	CMDIFrameWnd::AssertValid();
}

void CMainFrame::Dump(CDumpContext& dc) const
{
	CMDIFrameWnd::Dump(dc);
}

void CMainFrame::OnMenuExcute(UINT nID)
{
	if (nID >= 22000 && nID <= 25000)
	{
		nID = nID - ID_PLUGINS_CMD;
		m_pluginManager->RunPluginCommand(nID);
	}
	
	
}

void CMainFrame::OnToolBarExcute(UINT nID)
{
	if (nID >= 30000 && nID <= 45000)
	{
		nID = nID - ID_TOOLBAR_CMD;
		m_pluginManager->RunToolBarCommand(nID);
	}
}

void CMainFrame::AddButtonToToolBar()
{
	int temp = m_tmphIcon.size();

	//int size = m_pluginhIcon.size();
	//int count = 0;
	m_imageList.Create(16, 16, ILC_COLOR32, 0, 4);
	//m_pluginhIcon = m_pluginManager->m_pluginhIcon;
	//list<vector<HICON>>::iterator vbegin = m_pluginhIcon.begin();
	//list<vector<HICON>>::iterator vend   = m_pluginhIcon.end();
	
	for (int i=0; i<m_tmphIcon.size(); i++)
	{
		m_imageList.Add(m_tmphIcon.at(i));
	}

	CToolBarCtrl &ctrlBar = m_pluginToolbar.GetToolBarCtrl();
	ctrlBar.SetImageList(&m_imageList);
	//ctrlBar.SetButtonSize(CSize(16, 16));
	ctrlBar.SetButtonWidth(100, 80);

	TBBUTTON btn;
	/*int i=0;
	for (; i<size; i++)
	{
		for (int j=0; j<size1; j++)
		{
			btn.iBitmap = i*size1+j;
			btn.idCommand = ID_TOOLBAR_CMD + i*size1+j;
			btn.fsState = TBSTATE_ENABLED;
			btn.fsStyle = TBSTYLE_BUTTON;
			btn.dwData = 0;
			btn.iString = NULL;
			ctrlBar.AddButtons(1, &btn);
		}

		TBBUTTON btn1;
		btn1.iBitmap = 0;
		btn1.idCommand = 0;
		btn1.fsState = TBSTATE_ENABLED;
		btn1.fsStyle = TBSTYLE_SEP;
		btn1.dwData = 0;
		btn1.iString = NULL;
		ctrlBar.AddButtons(1, &btn1);
		
	}*/

	
	for (int j=0; j<m_tmphIcon.size(); j++)
	{
		if (j%5 == 0)
		{
			TBBUTTON btn1;
			btn1.iBitmap = 0;
			btn1.idCommand = 0;
			btn1.fsState = TBSTATE_ENABLED;
			btn1.fsStyle = TBSTYLE_SEP;
			btn1.dwData = 0;
			btn1.iString = NULL;
			ctrlBar.AddButtons(1, &btn1);
		}
		btn.iBitmap = j;
		btn.idCommand = ID_TOOLBAR_CMD + j;
		btn.fsState = TBSTATE_ENABLED;
		btn.fsStyle = TBSTYLE_BUTTON;
		btn.dwData = 0;
		btn.iString = NULL;
		ctrlBar.AddButtons(1, &btn);
	}

	/*TBBUTTON btn1;
	btn1.iBitmap = 0;
	btn1.idCommand = 0;
	btn1.fsState = TBSTATE_ENABLED;
	btn1.fsStyle = TBSTYLE_SEP;
	btn1.dwData = 0;
	btn1.iString = NULL;
	ctrlBar.AddButtons(1, &btn1);*/


	//RepositionBars(AFX_IDW_CONTROLBAR_FIRST,AFX_IDW_CONTROLBAR_LAST,0);

}
#endif //_DEBUG


// CMainFrame message handlers




BOOL CMainFrame::OnCommand(WPARAM wParam, LPARAM lParam)
{
	// TODO: Add your specialized code here and/or call the base class
	if (wParam >= 30000 && wParam <= 40000)
	{
		wParam = wParam-ID_TOOLBAR_CMD;
		m_pluginManager->RunToolBarCommand(wParam);
	}
	return CMDIFrameWnd::OnCommand(wParam, lParam);
}


//void CMainFrame::OnToolBT1()
//{
//	AfxMessageBox("1");
//}
//
//void CMainFrame::OnToolBT2()
//{
//	AfxMessageBox("2");
//}
//
//void CMainFrame::OnToolBT3()
//{
//	AfxMessageBox("3");
//}
//
//void CMainFrame::OnToolBT4()
//{
//	AfxMessageBox("4");
//}
//
//void CMainFrame::OnToolBT5()
//{
//	AfxMessageBox("5");
//}

