// ViewManagement.cpp : Defines the initialization routines for the DLL.
//

#include "stdafx.h"
#include "ViewManagement.h"
#include "ViewTest.h"
#ifdef _DEBUG
#define new DEBUG_NEW
#endif

//
//TODO: If this DLL is dynamically linked against the MFC DLLs,
//		any functions exported from this DLL which call into
//		MFC must have the AFX_MANAGE_STATE macro added at the
//		very beginning of the function.
//
//		For example:
//
//		extern "C" BOOL PASCAL EXPORT ExportedFunction()
//		{
//			AFX_MANAGE_STATE(AfxGetStaticModuleState());
//			// normal function body here
//		}
//
//		It is very important that this macro appear in each
//		function, prior to any calls into MFC.  This means that
//		it must appear as the first statement within the 
//		function, even before any object variable declarations
//		as their constructors may generate calls into the MFC
//		DLL.
//
//		Please see MFC Technical Notes 33 and 58 for additional
//		details.
//


// CViewManagementApp

BEGIN_MESSAGE_MAP(CViewManagementApp, CWinApp)
END_MESSAGE_MAP()


// CViewManagementApp construction

CViewManagementApp::CViewManagementApp()
{
	// TODO: add construction code here,
	// Place all significant initialization in InitInstance
}


// The one and only CViewManagementApp object

CViewManagementApp theApp;
CViewTest viewtest(" ”Õºπ‹¿Ì", 6);
std::vector<HICON> m_ToolBarIcon;

extern "C" __declspec(dllexport) char *GetName(void);
extern "C" __declspec(dllexport) MenuItem *GetMenuArray(int *nbF);
extern "C" __declspec(dllexport) ToolBarItem *GetToolBarArray(int *nbF);
extern "C" __declspec(dllexport) void GetPlugInIcon(vector<HICON> &hIcon);

// CViewManagementApp initialization

BOOL CViewManagementApp::InitInstance()
{
	CWinApp::InitInstance();
	m_ToolBarIcon.push_back(theApp.LoadIcon(IDI_ICON1));
	m_ToolBarIcon.push_back(theApp.LoadIcon(IDI_ICON2));
	m_ToolBarIcon.push_back(theApp.LoadIcon(IDI_ICON3));
	m_ToolBarIcon.push_back(theApp.LoadIcon(IDI_ICON4));
	m_ToolBarIcon.push_back(theApp.LoadIcon(IDI_ICON5));
	m_ToolBarIcon.push_back(theApp.LoadIcon(IDI_ICON6));
	//m_ToolBarIcon.push_back(theApp.LoadIcon(IDI_ICON7));
	//m_ToolBarIcon.push_back(theApp.LoadIcon(IDI_ICON8));
	return TRUE;
}


extern "C" __declspec(dllexport) char *GetName(void)
{
	return viewtest.GetPluginName();
}
extern "C" __declspec(dllexport) MenuItem *GetMenuArray(int *nbF)
{
	*nbF = viewtest.GetMenuItemCount();
	return viewtest.GetMenuItem();
}

extern "C" __declspec(dllexport) ToolBarItem *GetToolBarArray(int *nbF)
{
	*nbF = viewtest.GetToolBarCount();
	return viewtest.GetToolBarItem();
}

extern "C" __declspec(dllexport) void GetPlugInIcon(vector<HICON> &hIcon)
{
	 hIcon = m_ToolBarIcon;
//	return m_ToolBarIcon;
}