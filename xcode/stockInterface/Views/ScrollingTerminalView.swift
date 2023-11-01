//
//  ScrollingTerminalView.swift
//  stockInterface
//
//  Created by CHONK on 10/24/23.
//

import Cocoa

class ScrollingTerminalView: NSView {

    // MARK: Subviews
    
    weak var inputParentView: NSView!
    weak var inputField:      NSTextField!
    
    // MARK: Properties
    
    // MARK: Style
    
    // MARK: Init
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupView()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    // MARK: Setup View
    
    private func setupView() {
        wantsLayer = true
        layer?.backgroundColor = .black
//        userinteraction
        
        let inputParentView = NSView()
        inputParentView.wantsLayer = true
        inputParentView.layer?.backgroundColor = .InputBackground
        inputParentView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(inputParentView)
        NSLayoutConstraint.activate([
            inputParentView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 6),
            inputParentView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -6),
            inputParentView.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -6),
            inputParentView.heightAnchor.constraint(equalToConstant: 44)
        ])
        self.inputParentView = inputParentView
        
//        let inputField = NSTextField(frame: NSRect(x: 0, y: 0, width: 50, height: 50))
        let inputField = NSTextField()
        
//        inputField.isBezeled = false
//        inputField.placeholderString = "Bob is gay"
//        inputField.font = NSFont(name: "BrassMono-Regular", size: 20)
//        inputField.isEnabled = true
//        inputField.isSelectable = true
//        inputField.allowsEditingTextAttributes = true
//        inputField.delegate = self
//        inputField.stringValue = "Gay But Sex"
//        inputField.contentType = .
//        inputField.isEditable = true
        inputField.translatesAutoresizingMaskIntoConstraints = false
        inputParentView.addSubview(inputField)
        NSLayoutConstraint.activate([
            inputField.topAnchor.constraint(equalTo: inputParentView.topAnchor),
            inputField.leadingAnchor.constraint(equalTo: inputParentView.leadingAnchor),
            inputField.trailingAnchor.constraint(equalTo: inputParentView.trailingAnchor),
            inputField.bottomAnchor.constraint(equalTo: inputParentView.bottomAnchor)
        ])
        self.inputField = inputField
//        inputField.
        
        for family in NSFontManager.shared.availableFontFamilies {
            print("\(family)")
            for name in NSFontManager.shared.availableMembers(ofFontFamily: family) ?? [] {
                print("   \(name[0])")
            }
        }

    }
    
}
