// PlugInArchitectureDoc.h : interface of the CPlugInArchitectureDoc class
//


#pragma once


class CPlugInArchitectureDoc : public CDocument
{
protected: // create from serialization only
	CPlugInArchitectureDoc();
	DECLARE_DYNCREATE(CPlugInArchitectureDoc)

// Attributes
public:

// Operations
public:

// Overrides
public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);

// Implementation
public:
	virtual ~CPlugInArchitectureDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
};


