// PlugInArchitecture.h : main header file for the PlugInArchitecture application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols


// CPlugInArchitectureApp:
// See PlugInArchitecture.cpp for the implementation of this class
//

class CPlugInArchitectureApp : public CWinApp
{
public:
	CPlugInArchitectureApp();


// Overrides
public:
	virtual BOOL InitInstance();

// Implementation
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
};

extern CPlugInArchitectureApp theApp;