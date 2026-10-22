// ProjectManagemnent.h : main header file for the ProjectManagemnent DLL
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols
#include <vector>
#include "Project.h"

// CProjectManagemnentApp
// See ProjectManagemnent.cpp for the implementation of this class
//

class CProjectManagemnentApp : public CWinApp
{
public:
	CProjectManagemnentApp();
	 
// Overrides
public:
	virtual BOOL InitInstance();

	DECLARE_MESSAGE_MAP()
	
};
