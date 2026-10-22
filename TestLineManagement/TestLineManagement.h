// TestLineManagement.h : main header file for the TestLineManagement DLL
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols


// CTestLineManagementApp
// See TestLineManagement.cpp for the implementation of this class
//

class CTestLineManagementApp : public CWinApp
{
public:
	CTestLineManagementApp();

// Overrides
public:
	virtual BOOL InitInstance();

	DECLARE_MESSAGE_MAP()
};
