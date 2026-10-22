// ViewManagement.h : main header file for the ViewManagement DLL
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols


// CViewManagementApp
// See ViewManagement.cpp for the implementation of this class
//

class CViewManagementApp : public CWinApp
{
public:
	CViewManagementApp();

// Overrides
public:
	virtual BOOL InitInstance();

	DECLARE_MESSAGE_MAP()
};
