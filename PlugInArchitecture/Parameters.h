#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <vector>
#include "ShortCut.h"
using namespace std;

const char pMenuItemName[] = "MenuItem.xml";
const char pPlugInName[] = "PlugIn.xml";

struct PlugInConfigurationInfo
{
	char pName[MAX_PATH];
	char pPath[MAX_PATH];
	PlugInConfigurationInfo(const char *name, const char *path)
	{
		strcpy(pName, name);
		strcpy(pPath, path);
	};
};

class PluginParameters
{
public:
	static PluginParameters *GetInstance()
	{
		return pSelf;
	}
	void DestroyInstance(void);
	void SetIsNoPlugin(bool isNoPlugin) { noPlugin = isNoPlugin; };
	bool IsNoPlugins(void) {return noPlugin; };
	vector<PluginCmdShortcut> & GetPluginCommandList(void) {return pluginCommands; };
	vector<PluginCmdShortcut> & GetPluginCustomizedCmds(void) {return pluginCustomizedCmds; };
	const char *GetPluginPath(void) const {return nppPath; };
	const char *GetAppDataPath(void) const {return appdataNppDir; };
	//void LoadConfigurationFromXML(void);
	void LoadMenuItemFromXML(vector<std::string> &vMenuItem);
	void LoadPlugInsFromXML(vector<PlugInConfigurationInfo> &vPlugInItem);
	/*void GetMenuItem(vector<const char *> &vMenuItem);
	void GetPlugInItem(vector<PlugInConfigurationInfo> &vPlugInItem);*/

protected:
	PluginParameters();
	~PluginParameters(){};
	static PluginParameters *pSelf;
	vector<PluginCmdShortcut> pluginCommands;
	vector<PluginCmdShortcut>pluginCustomizedCmds;
	char shortcutsPath[MAX_PATH];
	char contextMenuPath[MAX_PATH];
	char nppPath[MAX_PATH];
	char appdataNppDir[MAX_PATH]; 
private:
	bool noPlugin;
	vector<std::string> m_MenuItems;
	vector<PlugInConfigurationInfo> m_PlugIn;

};
#endif