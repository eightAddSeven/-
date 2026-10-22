#include "stdafx.h"
#include "Parameters.h"
#include <assert.h>
#include "tinyxml.h"
#include <shlobj.h>
#include <shlwapi.h>

PluginParameters * PluginParameters::pSelf = new PluginParameters;

PluginParameters::PluginParameters()
{
	appdataNppDir[0] ='\0'; 
}

void PluginParameters::DestroyInstance()
{
	delete pSelf;
	pSelf = NULL;
}

void PluginParameters::LoadMenuItemFromXML(vector<std::string> &vMenuItem)
{
	//assert(pfileName != NULL);
	char ProjectPath[MAX_PATH];
	::GetModuleFileName(NULL, ProjectPath, sizeof(ProjectPath));
	PathRemoveFileSpec(ProjectPath);
	PathAppend(ProjectPath, pMenuItemName);
	TiXmlDocument doc(ProjectPath);
	doc.LoadFile();
	if (!doc.Parse(ProjectPath))
	{
		AfxMessageBox(doc.ErrorDesc());
		return;
	}

	const TiXmlElement *Root = doc.RootElement();
	for (const TiXmlNode *child = Root->FirstChild(); child; child = child->NextSibling())
	{
		if ((child->Type() == TiXmlNode::ELEMENT) && (!strcmp(child->Value(), "MenuItem")))
		{
			const TiXmlElement *childMenu = (const TiXmlElement *) child;
			const char *pItem;
			const char *pAttrTitle;
			pAttrTitle = childMenu->Attribute("title");
			for (const TiXmlNode *validateItem = childMenu->FirstChild(); validateItem; 
				validateItem = validateItem->NextSibling())
			{
				if (validateItem->Type() == TiXmlNode::ELEMENT)
				{
					const TiXmlElement *validateElement = (const TiXmlElement *)validateItem;
					if (!strcmp(validateElement->Value(), "Item"))
					{
						//pItem = validateElement->Attribute("name");
						vMenuItem.push_back(validateElement->Attribute("name"));
					}
				}
			}
		}
	}
}

void PluginParameters::LoadPlugInsFromXML(vector<PlugInConfigurationInfo> &vPlugInItem)
{
	//assert(pfileName != NULL);
	char ProjectPath[MAX_PATH];
	::GetModuleFileName(NULL, ProjectPath, sizeof(ProjectPath));
	PathRemoveFileSpec(ProjectPath);
	PathAppend(ProjectPath, pPlugInName);
	TiXmlDocument doc(ProjectPath);
	doc.LoadFile();
	if (!doc.Parse(ProjectPath))
	{
		AfxMessageBox(doc.ErrorDesc());
		return;
	}

	const TiXmlElement *Root = doc.RootElement();
	for (const TiXmlNode *child = Root->FirstChild(); child; child = child->NextSibling())
	{
		if ((child->Type() == TiXmlNode::ELEMENT) && (!strcmp(child->Value(), "PlugIn")))
		{
			const TiXmlElement *childMenu = (const TiXmlElement *) child;
			const char *pItem, *pURL;
			const char *pAttrTitle;
			pAttrTitle = childMenu->Attribute("title");
			for (const TiXmlNode *validateItem = childMenu->FirstChild(); validateItem; 
				validateItem = validateItem->NextSibling())
			{
				if (validateItem->Type() == TiXmlNode::ELEMENT)
				{
					const TiXmlElement *validateElement = (const TiXmlElement *)validateItem;
					if (!strcmp(validateElement->Value(), "Item"))
					{
						pItem = validateElement->Attribute("PlugInName");
						pURL  = validateElement->Attribute("url");
						PlugInConfigurationInfo Pci(pItem, pURL);
						vPlugInItem.push_back(Pci);
					}
				}
			}
		}
	}

}

//void PluginParameters::GetMenuItem(vector<const char *> &vMenuItem)
//{
//	/*LoadMenuItemFromXML();
//	vMenuItem = m_MenuItems;*/
//}
//
//void PluginParameters::GetPlugInItem(vector<PlugInConfigurationInfo> &vPlugInItem)
//{
//	/*LoadPlugInsFromXML();
//	vPlugInItem = m_PlugIn;*/
//}

//void Parameters::LoadConfigurationFromXML()
//{
//	LoadMenuItemF
//}