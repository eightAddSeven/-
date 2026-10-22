// ProjectManagemnent.cpp : Defines the initialization routines for the DLL.
//

#include "stdafx.h"
#include "ProjectManagemnent.h"

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


// CProjectManagemnentApp

BEGIN_MESSAGE_MAP(CProjectManagemnentApp, CWinApp)
END_MESSAGE_MAP()


// CProjectManagemnentApp construction

CProjectManagemnentApp::CProjectManagemnentApp()
{
	// TODO: add construction code here,
	// Place all significant initialization in InitInstance
}


// The one and only CProjectManagemnentApp object

CProjectManagemnentApp theApp;
std::vector<HICON> m_ToolBarIcon;
Project project("工程管理", 7);

extern "C" __declspec(dllexport) char *GetName(void);
extern "C" __declspec(dllexport) MenuItem *GetMenuArray(int *nbF);
extern "C" __declspec(dllexport) ToolBarItem *GetToolBarArray(int *nbF);
extern "C" __declspec(dllexport) void GetPlugInIcon(vector<HICON> &hIcon);
void InitIcon(void);
// CProjectManagemnentApp initialization

BOOL CProjectManagemnentApp::InitInstance()
{
	

	CWinApp::InitInstance();
	m_ToolBarIcon.push_back(theApp.LoadIcon(IDI_ICON1));
	m_ToolBarIcon.push_back(theApp.LoadIcon(IDI_ICON2));
	m_ToolBarIcon.push_back(theApp.LoadIcon(IDI_ICON3));
	m_ToolBarIcon.push_back(theApp.LoadIcon(IDI_ICON4));
	m_ToolBarIcon.push_back(theApp.LoadIcon(IDI_ICON5));
	m_ToolBarIcon.push_back(theApp.LoadIcon(IDI_ICON6));
	m_ToolBarIcon.push_back(theApp.LoadIcon(IDI_ICON7));
	return TRUE;
}

extern "C" __declspec(dllexport) char *GetName(void)
{
	return project.GetPluginName();
}

extern "C" __declspec(dllexport) MenuItem *GetMenuArray(int *nbF)
{
	*nbF = project.GetMenuItemCount();
	return project.GetMenuItem();
}

extern "C" __declspec(dllexport) ToolBarItem *GetToolBarArray(int *nbF)
{
	*nbF = project.GetMenuItemCount();
	return project.GetToolBarItem();
}

extern "C" __declspec(dllexport)void GetPlugInIcon(vector<HICON> &hIcon)
{
//	InitIcon();
	hIcon = m_ToolBarIcon;
	//return  m_ToolBarIcon;
}

void InitIcon(void)
{
	/*HICON hIcon1 = (HICON)::LoadImage(theApp, 
		MAKEINTRESOURCE(IDI_ICON1), IMAGE_ICON, 32, 32, 0);
	m_ToolBarIcon.push_back(hIcon1);

	HICON hIcon2 = (HICON)::LoadImage(hInstance, 
		MAKEINTRESOURCE(IDI_ICON1), IMAGE_ICON, 32, 32, 0);
	m_ToolBarIcon.push_back(hIcon2);

	HICON hIcon3 = (HICON)::LoadImage(hInstance, 
		MAKEINTRESOURCE(IDI_ICON1), IMAGE_ICON, 32, 32, 0);
	m_ToolBarIcon.push_back(hIcon3);

	HICON hIcon4 = (HICON)::LoadImage(hInstance, 
		MAKEINTRESOURCE(IDI_ICON1), IMAGE_ICON, 32, 32, 0);
	m_ToolBarIcon.push_back(hIcon4);

	HICON hIcon5 = (HICON)::LoadImage(hInstance, 
		MAKEINTRESOURCE(IDI_ICON1), IMAGE_ICON, 32, 32, 0);
	m_ToolBarIcon.push_back(hIcon5);

	HICON hIcon6 = (HICON)::LoadImage(hInstance, 
		MAKEINTRESOURCE(IDI_ICON1), IMAGE_ICON, 32, 32, 0);
	m_ToolBarIcon.push_back(hIcon6);

	HICON hIcon7 = (HICON)::LoadImage(hInstance, 
		MAKEINTRESOURCE(IDI_ICON1), IMAGE_ICON, 32, 32, 0);
	m_ToolBarIcon.push_back(hIcon7);*/
}