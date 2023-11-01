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
        
        let inputField = NSTextField()
        inputField.placeholderString = "Enter Command"
        inputField.font = NSFont(name: "BrassMono-Bold", size: 16)
        inputField.focusRingType = .none
        inputField.delegate = self
        inputField.resignFirstResponder()
        inputField.isBordered = false
        inputField.backgroundColor = .clear
        inputField.translatesAutoresizingMaskIntoConstraints = false
        inputParentView.addSubview(inputField)
        NSLayoutConstraint.activate([
            inputField.leadingAnchor.constraint(equalTo: inputParentView.leadingAnchor, constant: 10),
            inputField.trailingAnchor.constraint(equalTo: inputParentView.trailingAnchor, constant: -10),
            inputField.centerYAnchor.constraint(equalTo: inputParentView.centerYAnchor)
        ])
        self.inputField = inputField

    }
}

extension ScrollingTerminalView: NSTextFieldDelegate {
    func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
        
        if commandSelector == #selector(NSResponder.deleteBackward(_:)) {
            var text = inputField.stringValue
            text = String(text.dropLast())
            inputField.stringValue = text
        }
        
        if commandSelector == #selector(NSResponder.insertNewline(_:)) {
            inputField.stringValue = ""
        }
        
        return true
    }
}
