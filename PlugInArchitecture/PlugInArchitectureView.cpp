// PlugInArchitectureView.cpp : implementation of the CPlugInArchitectureView class
//

#include "stdafx.h"
#include "PlugInArchitecture.h"

#include "PlugInArchitectureDoc.h"
#include "PlugInArchitectureView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CPlugInArchitectureView

IMPLEMENT_DYNCREATE(CPlugInArchitectureView, CView)

BEGIN_MESSAGE_MAP(CPlugInArchitectureView, CView)
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CView::OnFilePrintPreview)
END_MESSAGE_MAP()

// CPlugInArchitectureView construction/destruction

CPlugInArchitectureView::CPlugInArchitectureView()
{
	// TODO: add construction code here

}

CPlugInArchitectureView::~CPlugInArchitectureView()
{
}

BOOL CPlugInArchitectureView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return CView::PreCreateWindow(cs);
}

// CPlugInArchitectureView drawing

void CPlugInArchitectureView::OnDraw(CDC* /*pDC*/)
{
	CPlugInArchitectureDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: add draw code for native data here
}


// CPlugInArchitectureView printing

BOOL CPlugInArchitectureView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void CPlugInArchitectureView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void CPlugInArchitectureView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}


// CPlugInArchitectureView diagnostics

#ifdef _DEBUG
void CPlugInArchitectureView::AssertValid() const
{
	CView::AssertValid();
}

void CPlugInArchitectureView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CPlugInArchitectureDoc* CPlugInArchitectureView::GetDocument() const // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CPlugInArchitectureDoc)));
	return (CPlugInArchitectureDoc*)m_pDocument;
}
#endif //_DEBUG


// CPlugInArchitectureView message handlers
