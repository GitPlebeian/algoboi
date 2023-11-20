//
//  AppDelegate.swift
//  stockInterface
//
//  Created by CHONK on 9/28/23.
//

import Cocoa

//@NSApplicationMain
class AppDelegate: NSObject, NSApplicationDelegate {

    var window: NSWindow!
    var isFullScreen = false
    var originalFrame: NSRect?
    var eventMonitor: Any?


    func applicationDidFinishLaunching(_ aNotification: Notification) {

        AllTickersController.shared.loadFromDisk()
        
        setupMenu()
        
        eventMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown]) { event in
            self.handleKeyDown(event: event)
            return event
        }


        NSApp.setActivationPolicy(.regular)

        let contentRect = NSRect(x: 0, y: 0, width: 1400, height: 800)
//        window = NSWindow(contentRect: contentRect, styleMask: [.borderless, .resizable], backing: .buffered, defer: false)
        window = NSWindow(contentRect: contentRect, styleMask: [.titled, .resizable], backing: .buffered, defer: false)
        window.hasShadow = true
        window.isMovableByWindowBackground = true
        window.isMovable = true
        window.isOpaque = false
//        window.isZoomable = true
//        window.set
//        window.contentView = LayoutManager.shared.getCelledMasterView()
        window.contentView = ContentView()
//        let contentRect = NSRect(x: 0, y: 0, width: NSScreen.main!.frame.width / 3, height: NSScreen.main!.frame.height - window.titlebarHeight)
//        window.setFrame(contentRect, display: true)
        window.backgroundColor = .clear
        window.makeKeyAndOrderFront(nil)
        window.center()

//        window.makeKeyAndOrderFront(nil)
    }
    
    func setupMenu() {
        let mainMenu = NSMenu()
        
        let appMenu = NSMenu()
        let appMenuItem = NSMenuItem()
        appMenuItem.submenu = appMenu
        appMenu.addItem(withTitle: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        mainMenu.addItem(appMenuItem)
        
        let viewMenu = NSMenu(title: "View")
        let viewMenuItem = NSMenuItem(title: "View", action: nil, keyEquivalent: "")
        
        let toggleFullscreenItem = NSMenuItem(title: "Toggle Fullscreen", action: #selector(toggleFullscreen), keyEquivalent: "f")
        toggleFullscreenItem.keyEquivalentModifierMask = [.command]
        
        viewMenu.addItem(toggleFullscreenItem)
        viewMenuItem.submenu = viewMenu
        mainMenu.addItem(viewMenuItem)
        
        NSApp.mainMenu = mainMenu
    }

    @objc func toggleFullscreen() {
        guard let screen = window.screen else { return }
        
        if isFullScreen {
            // Exit fullscreen mode
            if let originalFrame = originalFrame {
                window.setFrame(originalFrame, display: true, animate: true)
            }
        } else {
            // Enter fullscreen mode
            originalFrame = window.frame
            window.setFrame(screen.frame, display: true, animate: true)
        }
        
        isFullScreen.toggle()
    }

    func applicationWillTerminate(aNotification: NSNotification) {
        // Insert code here to tear down your application
        print("Bob")
    }
    
    func handleKeyDown(event: NSEvent) {
        EventController.shared.handleEvent(event: event)
    }

}

