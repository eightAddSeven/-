// PlugInArchitectureDoc.cpp : implementation of the CPlugInArchitectureDoc class
//

#include "stdafx.h"
#include "PlugInArchitecture.h"

#include "PlugInArchitectureDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CPlugInArchitectureDoc

IMPLEMENT_DYNCREATE(CPlugInArchitectureDoc, CDocument)

BEGIN_MESSAGE_MAP(CPlugInArchitectureDoc, CDocument)
END_MESSAGE_MAP()


// CPlugInArchitectureDoc construction/destruction

CPlugInArchitectureDoc::CPlugInArchitectureDoc()
{
	// TODO: add one-time construction code here

}

CPlugInArchitectureDoc::~CPlugInArchitectureDoc()
{
}

BOOL CPlugInArchitectureDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: add reinitialization code here
	// (SDI documents will reuse this document)

	return TRUE;
}




// CPlugInArchitectureDoc serialization

void CPlugInArchitectureDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: add storing code here
	}
	else
	{
		// TODO: add loading code here
	}
}


// CPlugInArchitectureDoc diagnostics

#ifdef _DEBUG
void CPlugInArchitectureDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CPlugInArchitectureDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CPlugInArchitectureDoc commands
