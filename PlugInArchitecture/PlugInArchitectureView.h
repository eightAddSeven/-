// PlugInArchitectureView.h : interface of the CPlugInArchitectureView class
//


#pragma once


class CPlugInArchitectureView : public CView
{
protected: // create from serialization only
	CPlugInArchitectureView();
	DECLARE_DYNCREATE(CPlugInArchitectureView)

// Attributes
public:
	CPlugInArchitectureDoc* GetDocument() const;

// Operations
public:

// Overrides
public:
	virtual void OnDraw(CDC* pDC);  // overridden to draw this view
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);

// Implementation
public:
	virtual ~CPlugInArchitectureView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
};

#ifndef _DEBUG  // debug version in PlugInArchitectureView.cpp
inline CPlugInArchitectureDoc* CPlugInArchitectureView::GetDocument() const
   { return reinterpret_cast<CPlugInArchitectureDoc*>(m_pDocument); }
#endif

